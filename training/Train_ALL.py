import csv
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score

import os

print("PreProcessing csv file")
def file_exists(file_path):
    """
    Check if a file exists at the specified path.
    """
    return os.path.exists(file_path)

def filter_csv(csv_file_path):
    """
    Read the CSV file, modify the file paths in the "files-changed" column,
    check if the modified file paths exist, and delete the rows if the files do not exist.
    """
    rows_to_keep = []
    rows_deleted = 0

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header
        rows_to_keep.append(header)  # Add header to the list of rows to keep
        for row in csv_reader:
            issue, file_path, y_value = row
            modified_file_path = file_path[:-5] + "-ALL.json"  # Modify the file path
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

# Example usage
csv_file_path = "dataset_ALL.csv"  # Replace with your CSV file path
filter_csv(csv_file_path)




# Load pre-trained BERT model and tokenizer
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read CSV file and extract data
print("Reading CSV file...")
csv_file_path = "dataset_ALL.csv"
issues = []
file_paths = []
y_values = []

with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row
    for i, row in enumerate(csv_reader):
        print(f"Reading row {i+1}...")
        issues.append(row[0])
        file_paths.append(row[1])
        y_values.append(int(row[2]))

# Vectorize issues using BERT
print("Vectorizing issues using BERT...")
vectorized_issues = []
for i, issue in enumerate(issues):
    print(f"Vectorizing issue {i+1}/{len(issues)}...")
    inputs = tokenizer(issue, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vector_representation = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    vectorized_issues.append(vector_representation)

# Read JSON files and extract vectorized data
print("Reading JSON files and extracting vectorized data...")
vectorized_java_code = []
max_vector_size = 0  # Initialize variable to keep track of the maximum vector size

for i, file_path in enumerate(file_paths):
    file_path_without_extension = file_path[:-5]
    print(f"Reading JSON file {i+1}/{len(file_paths)}: {file_path_without_extension}")
    json_file_path = file_path_without_extension + "-ALL.json"
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)
    vectorized_data_found = False
    vectors = []
    for key, value in json_data.items():
        if isinstance(value, list) and all(isinstance(x, float) for x in value):
            print(f"Vectorized data extracted for key: {key}")
            vectors.extend(value)  # Concatenate vectors for all tokens
            vectorized_data_found = True
    if not vectorized_data_found:
        raise ValueError(f"No vectorized data found in JSON file: {json_file_path}")
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

# Prepare data for model training
print("Preparing data for model training...")
X = np.concatenate((vectorized_issues, vectorized_java_code), axis=1)
y = np.array(y_values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert data to PyTorch tensors and create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define Transformer model for bug localization
class BugLocalizationTransformerModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BugLocalizationTransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# Define and train Transformer model
print("Defining and training Transformer model...")
Bmodel = BugLocalizationTransformerModel(input_size=X.shape[1], output_size=1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(Bmodel.parameters(), lr=0.0001)

for epoch in range(10):  # Adjust number of epochs as needed
    print(f"Epoch {epoch+1}/10")
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = Bmodel(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{len(train_loader)}: Loss {loss.item():.4f}")

print("Saving the trained model...")
torch.save(Bmodel.state_dict(), "Model_ALL.pth")

# Evaluate model
print("Evaluating model...")
with torch.no_grad():
    outputs = Bmodel(torch.tensor(X_test, dtype=torch.float))
    predicted_labels = (outputs > 0.5).squeeze().numpy()
accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy:", accuracy)

# Function to predict top files for an issue
def predict_top_files_for_issue(Bmodel, tokenizer, issue, file_paths, top_n=10):
    # Vectorize the issue
    inputs = tokenizer(issue, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vectorized_issue = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Concatenate issue vector with each Java file vector and predict probabilities
    concatenated_vectors = [np.concatenate((vectorized_issue, vectorized_java), axis=0) 
                            for vectorized_java in vectorized_java_code]
    concatenated_tensors = torch.tensor(concatenated_vectors, dtype=torch.float)
    with torch.no_grad():
        probabilities = Bmodel(concatenated_tensors).squeeze().numpy()

    # Sort files based on probabilities and return top n
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_files = [file_paths[i] for i in top_indices]
    top_probabilities = [probabilities[i] for i in top_indices]

    return top_files, top_probabilities

# Function to calculate MAP and MRR
def calculate_map_mrr(Bmodel, tokenizer, issues, file_paths, y_values):
    unique_issues = list(set(issues))
    all_aps = []
    all_rrs = []
    total_issues = len(unique_issues)
    
    for idx, issue_text in enumerate(unique_issues):
        print(f"Calculating MAP and MRR for issue {idx + 1}/{total_issues}: '{issue_text}'")
        relevant_files = [file_paths[i] for i, issue in enumerate(issues) if issue == issue_text and y_values[i] == 1]
        predicted_files, predicted_probabilities = predict_top_files_for_issue(Bmodel, tokenizer, issue_text, file_paths, top_n=200)

        # Calculate Average Precision
        ap = 0.0
        num_correct = 0
        for j, file in enumerate(predicted_files):
            if file in relevant_files:
                num_correct += 1
                ap += num_correct / (j + 1)
        if len(relevant_files)!=0:
            ap /= len(relevant_files)
            all_aps.append(ap)
        else:
            all_aps.append(0.0)

        # Calculate Reciprocal Rank
        rr = 0.0
        for j, file in enumerate(predicted_files):
            if file in relevant_files:
                rr = 1 / (j + 1)
                break
        all_rrs.append(rr)

    map_score = np.mean(all_aps)
    mrr_score = np.mean(all_rrs)
    return map_score, mrr_score

# Calculate MAP and MRR
map_score, mrr_score = calculate_map_mrr(Bmodel, tokenizer, issues, file_paths, y_values)
print("Mean Average Precision (MAP):", map_score)
print("Mean Reciprocal Rank (MRR):", mrr_score)

def predict_top_unique_files_for_issue(Bmodel, tokenizer, new_issue, file_paths, top_n=10):
    # Vectorize the new issue
    inputs = tokenizer(new_issue, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
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


def print_predicted_matches(Bmodel, tokenizer, issues, file_paths, y_values):
    unique_issues = list(set(issues))
    total_issues = len(unique_issues)
    
    for idx, issue_text in enumerate(unique_issues):
        print(f"Evaluating issue {idx + 1}/{total_issues}: '{issue_text}'")
        relevant_files = [file_paths[i] for i, issue in enumerate(issues) if issue == issue_text and y_values[i] == 1]
        predicted_files, _ = predict_top_unique_files_for_issue(Bmodel, tokenizer, issue_text, file_paths, top_n=15)

        # Calculate the number of predicted files that match the actual files changed
        num_matches = len(set(predicted_files) & set(relevant_files))

        print(f"Number of predicted files matching actual files changed: {num_matches}/{len(relevant_files)}")

# Print the number of predicted matches
print_predicted_matches(Bmodel, tokenizer, issues, file_paths, y_values)

# Example usage:
new_issue = "This is a new bug description"
top_files, top_probabilities = predict_top_unique_files_for_issue(Bmodel, tokenizer, new_issue, file_paths, top_n=10)
print("Top 10 unique files for the new issue:")
for file, probability in zip(top_files, top_probabilities):
    print(f"File: {file}")
