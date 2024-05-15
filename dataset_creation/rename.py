import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith("-PDG.json"):
            old_path = os.path.join(directory, filename)
            new_filename = filename.replace("-PDG.json", "-DFG.json")
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

# Replace 'directory_path' with the path to your directory containing the files
directory_path = "./hollow-master"
rename_files(directory_path)
