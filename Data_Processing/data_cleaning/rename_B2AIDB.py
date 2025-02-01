import os

# Source directory containing the files
source_dir = "/home/b/bhavyareddyseerapu/filtered_b2ai_data"

# Mapping for file suffixes based on specific keywords
SUFFIX_MAPPING = {
    "Respiration-and-cough_rec-Respiration-and-cough-Breath-1": "Reg1_Avid_NS",
    "Respiration-and-cough_rec-Respiration-and-cough-Breath-2": "Reg2_Avid_NS",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-2": "Deep1_Avid_NS",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-4": "Deep2_Avid_NS",
    "Respiration-and-cough_rec-Respiration-and-cough-ThreeQuickBreaths-1": "FIMO1_Avid_NS",
    "Respiration-and-cough_rec-Respiration-and-cough-ThreeQuickBreaths-2": "FIMO2_Avid_NS",
    "Rainbow-Passage_rec-Rainbow-Passage": "RP_Avid_NS"
}

def rename_files(src):
    for root, _, files in os.walk(src):
        for file in files:
            # Check if the file is a .wav file
            if not file.lower().endswith(".wav"):
                continue
            
            # Process each mapping to identify the appropriate suffix
            for key, suffix in SUFFIX_MAPPING.items():
                if key in file:
                    # Extract the prefix (sub-ID and ses-ID)
                    prefix_parts = file.split("_")
                    sub_id = prefix_parts[0]  # sub-ID
                    ses_id = prefix_parts[1]  # ses-ID
                    
                    # Construct the new filename with the suffix
                    new_name = f"{sub_id}_{ses_id}_{suffix}.wav"
                    
                    # Rename the file
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, new_name)
                    os.rename(old_file_path, new_file_path)
                    
                    print(f"Renamed: {old_file_path} -> {new_file_path}")
                    break

# Run the rename function
rename_files(source_dir)

print("Renaming completed.")
