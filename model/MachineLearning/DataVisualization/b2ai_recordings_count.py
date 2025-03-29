import os
import glob
from collections import defaultdict
import re

def count_recordings(base_dir, subjects_list):
    """
    Count recordings for specified subjects from a directory structure
    Args:
        base_dir (str): Base directory containing the subject folders
        subjects_list (list): List of subject IDs to count
    Returns:
        dict: Counts of recordings by type
    """
    # Initialize counts
    counts = {
        "Subject_count": len(subjects_list),
        "FIMO": 0,
        "Reg": 0,
        "RP": 0,
        "Deep": 0,
        "Total": 0
    }
    
    # Process each subject
    for subject in subjects_list:
        subject_dir = os.path.join(base_dir, subject)
        
        # Skip if subject directory doesn't exist
        if not os.path.isdir(subject_dir):
            print(f"Warning: Directory not found for {subject}")
            continue
        
        # Find all audio files for this subject
        pattern = os.path.join(subject_dir, "ses-*", "audio", f"{subject}_ses-*_*.wav")
        audio_files = glob.glob(pattern)
        
        for file_path in audio_files:
            counts["Total"] += 1
            
            # Extract recording type using regex
            # Pattern looks for FIMO, Reg, RP, or Deep in the filename
            # Including numbered variants like FIMO1, FIMO2, Reg1, Reg2, Deep1, Deep2
            match = re.search(r'_(FIMO\d*|Reg\d*|RP|Deep\d*)_', os.path.basename(file_path))
            if match:
                recording_type = match.group(1)
                
                # Group all variants to their parent category
                if recording_type.startswith('FIMO'):
                    counts["FIMO"] += 1
                elif recording_type.startswith('Reg'):
                    counts["Reg"] += 1
                elif recording_type.startswith('Deep'):
                    counts["Deep"] += 1
                elif recording_type == 'RP':
                    counts["RP"] += 1
    
    return counts

# List of subjects to process
subjects = [
    'sub-ec0d9ed5-0083-4ff7-af85-7ddcc1e5142c',
    'sub-daa7c187-8f36-4b2a-a4bc-7d2c3f6f09b2',
    'sub-f9988983-ebb5-434c-a20e-461e77f24cad',
    'sub-bcf75f58-596e-4f20-b78f-dfdd70a6c748',
    'sub-8896b265-55d5-4712-86ba-29b9090c5a9d',
    'sub-a33afb95-ab9b-4729-af0d-c64649f63669',
    'sub-7a23e78c-b42b-4de4-b0a5-771fce839c75',
    'sub-e96ef2c0-eb50-4b1d-afb5-81aaf9a1f2df',
    'sub-3132b52d-a9f8-4ebe-bf19-bab838f8c96b',
    'sub-d46aa720-946b-4e5d-b05d-ab5a97a9dbc6',
    'sub-8d5dc52b-e8aa-42e7-ae54-8f05c4667d39',
    'sub-ca1b69b7-444d-411f-861b-7ff00b4eb9fa',
    'sub-149f5b8a-aa7e-4806-9025-606c9fac95a2',
    'sub-62485abf-d6cd-45f7-bd4f-70f0621ee640',
    'sub-a2044905-fcf0-4971-8567-7d5a772f431c',
    'sub-48984210-19a7-4b56-abd2-401f7dbfdf31'
]

# Base directory where all subject folders are located
base_directory = '/home/b/bhavyareddyseerapu/filtered_b2ai_data'

# Run the analysis
results = count_recordings(base_directory, subjects)

# Print the results
print("Summary of Recordings:")
print("---------------------")
print(f"Subject count: {results['Subject_count']}")
print(f"FIMO recordings: {results['FIMO']}")
print(f"Reg recordings: {results['Reg']}")
print(f"RP recordings: {results['RP']}")
print(f"Deep recordings: {results['Deep']}")
print(f"Total recordings: {results['Total']}")