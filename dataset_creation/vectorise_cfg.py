import os
import json
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def process_gml_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Tokenize the text
    tokens = [simple_preprocess(sentence) for sentence in text.split('\n')]

    # Train Word2Vec model
    model = Word2Vec(tokens, vector_size=10, window=5, min_count=1, workers=4)

    # Prepare dictionary to store word vectors
    word_vectors = {}
    for word in model.wv.index_to_key:
        word_vectors[word] = model.wv[word].tolist()

    # Write word vectors to JSON file
    output_file = os.path.splitext(file_path)[0] + '.json'
    with open(output_file, 'w') as json_file:
        json.dump(word_vectors, json_file)

    print(f"Word vectors for {file_path} have been saved to {output_file}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("CFG.gml"):
                file_path = os.path.join(root, file)
                process_gml_file(file_path)

# Replace 'directory_path' with the path to your directory containing .gml files
directory_path = './hollow-master/'  # Current directory
process_directory(directory_path)

