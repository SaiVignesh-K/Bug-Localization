#!/bin/bash

# Define the directory containing your Java files
java_directory="./hollow-master/"

# Traverse through the directory and its subdirectories
find "$java_directory" -type f -name "*.java" | while read -r java_file; do
    # Extract the filename without extension
    java_filename=$(basename -- "$java_file")
    java_filename_no_ext="${java_filename%.*}"
    
    # Extract the directory path of the Java file
    java_dir=$(dirname "$java_file")

    # Run comex command to generate DOT files
    comex --lang "java" --code-file "$java_file" --graphs "ast,cfg,dfg"

    # Move the output DOT file to the same directory as the Java file
    mv "output.dot" "$java_dir/$java_filename_no_ext-ALL.dot"

    echo "Converted $java_filename to $java_dir/$java_filename_no_ext-ALL.dot"
done

