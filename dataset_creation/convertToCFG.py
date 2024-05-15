import os
import subprocess

# Define the directory containing Java files
java_directory = "./hollow-master"

# Path to Progex JAR file
progex_jar = "PROGEX.jar"

# Iterate over Java files in the directory
for root, dirs, files in os.walk(java_directory):
    for file in files:
        if file.endswith('.java'):
            java_file_path = os.path.join(root, file)
            cfg_output_path = os.path.splitext(java_file_path)[0] + '.gml'
            
            try:
                # Generate CFG using Progex and store it beside the Java file
                subprocess.run(['java', '-jar', progex_jar, '-cfg', '-outdir', root, '-format', 'GML', java_file_path])
                print(f"Generated PDG for {java_file_path}")
            except Exception as e:
                print(f"Error processing {java_file_path}: {str(e)}")
                continue

print("Conversion completed.")
