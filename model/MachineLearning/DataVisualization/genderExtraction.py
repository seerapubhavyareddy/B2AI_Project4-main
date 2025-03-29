import json
import os
import pandas as pd

# Function to extract subject and gender from JSON file
def extract_subject_and_gender(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    # Extracting subject ID and gender from the JSON structure
    subject_id = None
    gender_identity = None
    
    for item in data.get('item', []):
        if item['linkId'] == 'record_id':
            subject_id = item['answer'][0]['valueString']
        elif item['linkId'] == 'gender_identity':
            gender_identity = item['answer'][0]['valueString']
    
    return subject_id, gender_identity

# Directory where the JSON files are stored
root_dir = '/home/b/bhavyareddyseerapu/bids_with_sensitive_recordings'

# List to store subject data
subject_data = []

# Walk through the directory to find all JSON files
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('_qgenericdemographicsschema.json'):
            json_file = os.path.join(subdir, file)
            subject_id, gender_identity = extract_subject_and_gender(json_file)
            if subject_id and gender_identity:
                # Append data to the list
                subject_data.append({'subject_id': subject_id, 'gender': gender_identity})

# Convert the data to a DataFrame
df = pd.DataFrame(subject_data)

# Remove duplicate rows based on subject_id, keeping the first one
df = df.drop_duplicates(subset=['subject_id'])

# Save to Excel
excel_path = 'subject_gender_data.xlsx'
df.to_excel(excel_path, index=False)

print(f"Data saved to {excel_path}")
