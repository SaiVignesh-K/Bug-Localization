import os
import json
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def process_dot_file(dot_file_path):
    with open(dot_file_path, 'r') as file:
        dot_content = file.read()

    # Extracting node and edge labels from .dot content
    node_labels = []
    edge_labels = []
    lines = dot_content.split('\n')
    for line in lines:
        if 'label=' in line:
            label_start = line.find('label="') + len('label="')
            label_end = line.find('"', label_start)
            label = line[label_start:label_end]
            node_labels.extend(label.split())
        elif 'label=' not in line:
            label_start = line.find('label=') + len('label=')
            label_end = line.find(']', label_start)
            label = line[label_start:label_end]
            edge_labels.append(label)

    # Tokenize the node and edge labels
    node_tokens = [simple_preprocess(label) for label in node_labels]
    edge_tokens = [simple_preprocess(label) for label in edge_labels]

    # Train Word2Vec model
    model = Word2Vec(node_tokens + edge_tokens, vector_size=10, window=5, min_count=1, workers=4)

    # Prepare dictionary to store word vectors
    word_vectors = {}
    for word in model.wv.index_to_key:
        word_vectors[word] = model.wv[word].tolist()

    # Write word vectors to JSON file
    output_file = os.path.splitext(dot_file_path)[0] + '.json'
    with open(output_file, 'w') as json_file:
        json.dump(word_vectors, json_file)

    print(f"Word vectors for {dot_file_path} have been saved to {output_file}")

def process_dot_files_in_directory(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('-ALL.dot'):
                dot_file_path = os.path.join(dirpath, filename)
                process_dot_file(dot_file_path)

# Assuming the directory containing subfolders is "./my_directory"
directory = "./hollow-master/"
process_dot_files_in_directory(directory)

