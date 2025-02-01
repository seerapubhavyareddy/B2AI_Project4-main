import os
import shutil

# Source and destination directories
source_dir = "/home/b/bhavyareddyseerapu/bids_with_sensitive_recordings"
dest_dir = "/home/b/bhavyareddyseerapu/filtered_b2ai_data"

# List of keywords to filter audio files
KEYWORDS = [
    "Respiration-and-cough_rec-Respiration-and-cough-Breath-1",
    "Respiration-and-cough_rec-Respiration-and-cough-Breath-2",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-2",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-4",
    "Respiration-and-cough_rec-Respiration-and-cough-ThreeQuickBreaths-1",
    "Respiration-and-cough_rec-Respiration-and-cough-ThreeQuickBreaths-2",
    "Rainbow-Passage_rec-Rainbow-Passage"
]

# Normalize keywords to lower case for case-insensitive matching
KEYWORDS = [keyword.lower() for keyword in KEYWORDS]

def should_copy_file(filename):
    """
    Check if the file should be copied based on the keywords and its extension.
    """
    if not filename.lower().endswith(".wav"):
        return False
    return any(keyword in filename.lower() for keyword in KEYWORDS)

def copy_filtered_files(src, dst):
    """
    Traverse the source directory, filter files, and copy them to the destination.
    """
    for root, dirs, files in os.walk(src):
        for file in files:
            if should_copy_file(file):
                # Get relative path to preserve folder structure
                rel_path = os.path.relpath(root, src)
                dest_folder = os.path.join(dst, rel_path)
                os.makedirs(dest_folder, exist_ok=True)  # Create destination folder if it doesn't exist

                # Copy file to the destination
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_folder, file)
                shutil.copy2(src_file, dest_file)
                # print(f"Copied: {src_file} -> {dest_file}")

# Create the destination directory
os.makedirs(dest_dir, exist_ok=True)

# Run the function
copy_filtered_files(source_dir, dest_dir)

print("File filtering and copying completed.")
