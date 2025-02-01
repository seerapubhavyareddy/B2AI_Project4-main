import os

# Directories to process
directories = [
    "/home/b/bhavyareddyseerapu/chunk_data/chunk_stridor",
    "/home/b/bhavyareddyseerapu/chunk_data/chunk_b2ai"
]

# Output files for each category
output_files = {
    "Deep": "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/Data_PreProcessing/deep_combined_files.txt",
    "Reg": "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/Data_PreProcessing/reg_combined_files.txt",
    "RP": "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/Data_PreProcessing/rp_combined_files.txt",
    "FIMO": "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/Data_PreProcessing/fimo_combined_files.txt",
}

# Clear the content of output files if they already exist
for file in output_files.values():
    open(file, 'w').close()

# Function to map variations to standard categories
def map_to_category(filename):
    if "Deep" in filename or "Deep1" in filename or "Deep2" in filename:
        return "Deep"
    elif "Reg" in filename or "Reg1" in filename or "Reg2" in filename:
        return "Reg"
    elif "FIMO" in filename or "FIMO1" in filename or "FIMO2" in filename:
        return "FIMO"
    elif "RP" in filename:
        return "RP"
    else:
        return None

# Function to process a directory and append data to output files
def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                # Get the full file path
                full_path = os.path.join(root, file)

                # Map the file to a category
                category = map_to_category(file)
                if category:
                    # Determine the label based on "NS" or "S"
                    label = "0" if "_NS_" in file else "1"
                    
                    # Write to the respective file
                    with open(output_files[category], "a") as out_file:
                        out_file.write(f"{full_path} {label}\n")

# Process both directories
for directory in directories:
    process_directory(directory)

print("Files have been consolidated into category-specific .txt files.")
