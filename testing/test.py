import csv
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import warnings
import torch.nn as nn
import os

#print("PreProcessing csv file")
def file_exists(file_path):
    """
    Check if a file exists at the specified path.
    """
    return os.path.exists(file_path)

def filter_csv(csv_file_path,v):
    """
    Read the CSV file, modify the file paths in the "files-changed" column,
    check if the modified file paths exist, and delete the rows if the files do not exist.
    """
    rows_to_keep = []
    rows_deleted = 0
    p=""
    if(v==1):
        p=".java-AST.json"
    elif(v==2):
        p="-CFG.json"
    elif(v==3):
        p="-DFG.json"
    elif(v==4):
        p="-Comb1.json"
    elif(v==5):
        p="-Comb2.json"
    elif(v==6):
        p="-Comb3.json"
    else:
        p="-ALL.json"            
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header
        rows_to_keep.append(header)  # Add header to the list of rows to keep
        for row in csv_reader:
            issue, file_path, y_value = row
            modified_file_path = file_path[:-5] +p  # Modify the file path
            if file_exists(modified_file_path):
                rows_to_keep.append(row)
            else:
                #print(f"File not found: {modified_file_path}. Deleting row...")
                rows_deleted += 1

    # Write the filtered rows back to the CSV file
    with open(csv_file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(rows_to_keep)

    #print(f"Deleted {rows_deleted} rows from the CSV file.")

class BugLocalizationTransformerModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BugLocalizationTransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return torch.sigmoid(x)

def load_model(model_path, input_size):
    model = BugLocalizationTransformerModel(input_size=input_size, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_top_files_for_issue(bertmodel, Bmodel, tokenizer, new_issue, file_paths,vectorized_java_code, top_n=10):
    # Vectorize the new issue
    inputs = tokenizer(new_issue, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = bertmodel(**inputs)
    vectorized_issue = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Concatenate new issue vector with each Java file vector and predict probabilities
    concatenated_vectors = [np.concatenate((vectorized_issue, vectorized_java), axis=0) 
                            for vectorized_java in vectorized_java_code]
    concatenated_tensors = torch.tensor(concatenated_vectors, dtype=torch.float)
    with torch.no_grad():
        probabilities = Bmodel(concatenated_tensors).squeeze().numpy()

    # Sort files based on probabilities and return top n unique files
    unique_file_probabilities = {}
    for i, probability in enumerate(probabilities):
        if file_paths[i] not in unique_file_probabilities:
            unique_file_probabilities[file_paths[i]] = probability
    top_unique_files = sorted(unique_file_probabilities, key=unique_file_probabilities.get, reverse=True)[:top_n]
    top_unique_probabilities = [unique_file_probabilities[file] for file in top_unique_files]

    return top_unique_files, top_unique_probabilities

def main(issue, model_type):

    csv_file_path = "mod.csv"  # Replace with your CSV file path
    filter_csv(csv_file_path,model_type)
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load CSV data and filter
    csv_file_path = "mod.csv"
    warnings.filterwarnings("ignore")
    issues, file_paths, y_values = [], [], []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  
        for row in csv_reader:
            issues.append(row[0])
            file_paths.append(row[1])
            y_values.append(int(row[2]))

    # Load pre-trained BERT model
    bertmodel = BertModel.from_pretrained('bert-base-uncased')
    print("Working on it...")
    print("Loading the saved model...")

    # Vectorize issues using BERT
    vectorized_issues = []
    for issue in issues:
        inputs = tokenizer(issue, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = bertmodel(**inputs)
        vector_representation = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        vectorized_issues.append(vector_representation)

    # Load vectorized Java code
    max_vector_size = 0 
    vectorized_java_code = []
    v=model_type
    p=""
    if(v==1):
        p=".java-AST.json"
    elif(v==2):
        p="-CFG.json"
    elif(v==3):
        p="-DFG.json"
    elif(v==4):
        p="-Comb1.json"
    elif(v==5):
        p="-Comb2.json"
    elif(v==6):
        p="-Comb3.json"
    else:
        p="-ALL.json" 
    for file_path in file_paths:
        file_path_without_extension = file_path[:-5]
        json_file_path = file_path_without_extension + p
        with open(json_file_path, "r") as json_file:
            json_data = json.load(json_file)
        vectors = []
        for key, value in json_data.items():
            if isinstance(value, list) and all(isinstance(x, float) for x in value):
                vectors.extend(value)
        vectorized_java_code.append(vectors)
        max_vector_size = max(max_vector_size, len(vectors))  # Update maximum vector size
        
    # Pad or truncate vectors to match the size of the largest vector
    for i, vectors in enumerate(vectorized_java_code):
        if len(vectors) < max_vector_size:
            # Pad vectors with zeros
            vectorized_java_code[i] += [0.0] * (max_vector_size - len(vectors))
        elif len(vectors) > max_vector_size:
            # Truncate vectors
            vectorized_java_code[i] = vectors[:max_vector_size]

    # Prepare data for model prediction
    X = np.concatenate((vectorized_issues, vectorized_java_code), axis=1)

    # Load the corresponding model
    z=""
    if(v==1):
        z="Model_AST.pth"
    elif(v==2):
        z="Model_CFG.pth"
    elif(v==3):
        z="Model_DFG.pth"
    elif(v==4):
        z="Model_AST_CFG.pth"
    elif(v==5):
        z="Model_AST_DFG.pth"
    elif(v==6):
        z="Model_CFG_DFG.pth"
    else:
        z="Model_ALL.pth"
    model_path = z
    input_size = X.shape[1]
    Bmodel = load_model(model_path, input_size)
    print("Testing the model with given issue...")

    # Predict top files for the issue
    top_files, top_probabilities = predict_top_files_for_issue(bertmodel,Bmodel, tokenizer, issue, file_paths, vectorized_java_code)

    print("Top 10 most probable buggy files for the given issue:")
    for file, probability in zip(top_files, top_probabilities):
        print(f"File: {file}")

if __name__ == "__main__":
    issue = input("Enter the issue description: ")
    model_type = int(input("Enter the model type:\n 1:ast\n 2:cfg\n 3:dfg\n 4:ast+cfg\n 5:ast+dfg\n 6:cfg+dfg\n 7:ast+cfg+dfg\n"))
    main(issue, model_type)
