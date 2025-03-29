#!/usr/bin/env python3
"""
Script to extract ages from participant JSON files and create a table
showing the distribution of Stridor and Control subjects across age groups.
"""
import os
import json
import csv
import sys

# Define paths
input_dir = '/home/b/bhavyareddyseerapu/bids_with_sensitive_recordings/'
output_dir = '/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/dataDistrubution_figures/'

# List of Stridor subjects (provided)
stridor_subjects = [
    "sub-ec0d9ed5-0083-4ff7-af85-7ddcc1e5142c",
    "sub-daa7c187-8f36-4b2a-a4bc-7d2c3f6f09b2",
    "sub-f9988983-ebb5-434c-a20e-461e77f24cad",
    "sub-bcf75f58-596e-4f20-b78f-dfdd70a6c748",
    "sub-8896b265-55d5-4712-86ba-29b9090c5a9d",
    "sub-a33afb95-ab9b-4729-af0d-c64649f63669",
    "sub-7a23e78c-b42b-4de4-b0a5-771fce839c75",
    "sub-e96ef2c0-eb50-4b1d-afb5-81aaf9a1f2df",
    "sub-3132b52d-a9f8-4ebe-bf19-bab838f8c96b",
    "sub-d46aa720-946b-4e5d-b05d-ab5a97a9dbc6",
    "sub-8d5dc52b-e8aa-42e7-ae54-8f05c4667d39",
    "sub-ca1b69b7-444d-411f-861b-7ff00b4eb9fa",
    "sub-149f5b8a-aa7e-4806-9025-606c9fac95a2",
    "sub-62485abf-d6cd-45f7-bd4f-70f0621ee640",
    "sub-a2044905-fcf0-4971-8567-7d5a772f431c",
    "sub-48984210-19a7-4b56-abd2-401f7dbfdf31"
]

# Create output directory if it doesn't exist
try:
    os.makedirs(output_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating output directory: {e}")
    sys.exit(1)

# Function to extract age from a participant JSON file
def extract_age_from_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Look for age in the item list
        for item in data.get('item', []):
            if item.get('linkId') == 'age' and 'answer' in item:
                for answer in item.get('answer', []):
                    if 'valueString' in answer:
                        try:
                            age = float(answer['valueString'])
                            return age
                        except ValueError:
                            print(f"Warning: Could not convert age to float: {answer['valueString']} in {file_path}")
                            return None
        
        print(f"Warning: Age not found in {file_path}")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to determine age group
def get_age_group(age):
    if age is None:
        return "N/A"
    elif age < 18:
        return "0-18"
    elif age < 35:
        return "18-34"
    elif age < 50:
        return "35-49"
    elif age < 65:
        return "50-64"
    else:
        return "65+"

# Main function
def main():
    print(f"Starting age extraction from: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Get list of subject directories
    try:
        subject_dirs = [d for d in os.listdir(input_dir) if d.startswith('sub-')]
        print(f"Found {len(subject_dirs)} subject directories")
    except Exception as e:
        print(f"Error listing input directory: {e}")
        sys.exit(1)
    
    # Extract ages from all participant files
    age_data = []
    
    for subject_dir in subject_dirs:
        subject_path = os.path.join(input_dir, subject_dir)
        
        if os.path.isdir(subject_path):
            # Look for participant JSON file
            try:
                participant_files = [f for f in os.listdir(subject_path) if f.endswith('_participant.json')]
            except Exception as e:
                print(f"Error listing files in {subject_path}: {e}")
                continue
            
            for participant_file in participant_files:
                file_path = os.path.join(subject_path, participant_file)
                age = extract_age_from_json(file_path)
                
                # Determine if this is a Stridor subject or Control
                group = "Stridor" if subject_dir in stridor_subjects else "True Control"
                
                age_data.append({
                    'subject_id': subject_dir,
                    'age': age,
                    'age_group': get_age_group(age),
                    'group': group
                })
    
    print(f"Successfully extracted ages from {len(age_data)} participants")
    
    # Save raw age data to CSV
    raw_csv_path = os.path.join(output_dir, 'participant_ages_with_groups.csv')
    try:
        with open(raw_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subject_id', 'age', 'age_group', 'group'])
            writer.writeheader()
            writer.writerows(age_data)
        print(f"Saved raw age data to {raw_csv_path}")
    except Exception as e:
        print(f"Error saving raw age data: {e}")
    
    # Create age distribution table
    age_groups = ["0-18", "18-34", "35-49", "50-64", "65+", "N/A"]
    groups = ["True Control", "Stridor"]
    
    # Initialize distribution dictionary
    distribution = {group: {age_group: 0 for age_group in age_groups} for group in groups}
    
    # Count participants in each category
    for data in age_data:
        group = data['group']
        age_group = data['age_group']
        distribution[group][age_group] += 1
    
    # Create table for CSV output
    table_data = []
    for group in groups:
        row = {'Group': group}
        for age_group in age_groups:
            row[age_group] = distribution[group][age_group]
        table_data.append(row)
    
    # Save distribution table to CSV
    table_csv_path = os.path.join(output_dir, 'age_group_distribution_table.csv')
    try:
        with open(table_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Group'] + age_groups)
            writer.writeheader()
            writer.writerows(table_data)
        print(f"Saved age distribution table to {table_csv_path}")
    except Exception as e:
        print(f"Error saving age distribution table: {e}")
    
    # Print distribution table to console
    print("\nAge Distribution Table:")
    print("-" * 80)
    header = "| {:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |".format("Group", "0-18", "18-34", "35-49", "50-64", "65+", "N/A")
    print(header)
    print("-" * 80)
    
    for group in groups:
        row = "| {:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |".format(
            group,
            distribution[group]["0-18"],
            distribution[group]["18-34"],
            distribution[group]["35-49"],
            distribution[group]["50-64"],
            distribution[group]["65+"],
            distribution[group]["N/A"]
        )
        print(row)
    
    print("-" * 80)
    
    print("\nAnalysis complete. All files saved to:", output_dir)

if __name__ == "__main__":
    main()