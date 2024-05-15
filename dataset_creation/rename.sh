#!/bin/bash

# Define the directory containing your Java files
java_directory="./hollow-master/"

# Traverse through the directory and its subdirectories
find "$java_directory" -type f -name "*-PDG.json" | while read -r java_file; do
    # Use rename command to replace -PDG.json with -DFG.json
    new_file=$(echo "$java_file" | sed 's/-PDG\.json$/-DFG.json/')
    mv "$java_file" "$new_file"
    echo "Renamed: $java_file to $new_file"
done

