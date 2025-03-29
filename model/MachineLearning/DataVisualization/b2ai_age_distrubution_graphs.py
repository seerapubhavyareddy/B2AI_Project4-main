import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define paths
input_dir = '/home/b/bhavyareddyseerapu/bids_with_sensitive_recordings/'
output_dir = '/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/dataDistrubution_figures/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

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

# For testing with a small set of files, uncomment this function
def test_extraction_on_sample():
    print("Running test extraction on a sample file...")
    sample_path = os.path.join(input_dir, 'sub-0ee1e1e1-0e86-42cc-9e9d-2cafd9f1e01c', 
                                'sub-0ee1e1e1-0e86-42cc-9e9d-2cafd9f1e01c_participant.json')
    if os.path.exists(sample_path):
        age = extract_age_from_json(sample_path)
        print(f"Extracted age from sample: {age}")
    else:
        print(f"Sample file not found: {sample_path}")
        
# Uncomment to test on just one file
# test_extraction_on_sample()

# Get list of subject directories
subject_dirs = [d for d in os.listdir(input_dir) if d.startswith('sub-')]

# Extract ages from all participant files
age_data = []

for subject_dir in subject_dirs:
    subject_path = os.path.join(input_dir, subject_dir)
    
    if os.path.isdir(subject_path):
        # Look for participant JSON file
        participant_files = [f for f in os.listdir(subject_path) if f.endswith('_participant.json')]
        
        for participant_file in participant_files:
            subject_id = subject_dir
            file_path = os.path.join(subject_path, participant_file)
            age = extract_age_from_json(file_path)
            
            if age is not None:
                age_data.append({
                    'subject_id': subject_id,
                    'age': age
                })

# Create DataFrame
df = pd.DataFrame(age_data)

# Save raw age data to CSV
raw_csv_path = os.path.join(output_dir, 'participant_ages.csv')
df.to_csv(raw_csv_path, index=False)
print(f"Saved raw age data to {raw_csv_path}")

# Create age groups
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# Create distribution table
age_distribution = df['age_group'].value_counts().sort_index().reset_index()
age_distribution.columns = ['Age Group', 'Count']

# Calculate percentages
total_subjects = len(df)
age_distribution['Percentage'] = (age_distribution['Count'] / total_subjects * 100).round(1)

# Save age distribution to CSV
distribution_csv_path = os.path.join(output_dir, 'age_distribution.csv')
age_distribution.to_csv(distribution_csv_path, index=False)
print(f"Saved age distribution to {distribution_csv_path}")

# Create a simple text table for direct console output
print("\nAge Distribution Table:")
print("-" * 40)
print(f"{'Age Group':<10} | {'Count':<7} | {'Percentage':<10}")
print("-" * 40)
for _, row in age_distribution.iterrows():
    print(f"{row['Age Group']:<10} | {row['Count']:<7} | {row['Percentage']:<10.1f}%")
print("-" * 40)
print(f"Total: {total_subjects} participants")

# If the matplotlib/seaborn dependencies are available, try to create visualizations
try:
    # 1. Histogram of ages
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'], bins=10, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Participant Ages', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_histogram.png'), dpi=300)
    plt.close()

    # 2. Bar chart of age groups
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Age Group', y='Count', data=age_distribution)
    plt.title('Distribution by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_group_bar.png'), dpi=300)
    plt.close()

    # 3. Pie chart of age groups
    plt.figure(figsize=(10, 10))
    plt.pie(age_distribution['Count'], labels=age_distribution['Age Group'], 
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Age Group Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_group_pie.png'), dpi=300)
    plt.close()

    # 4. Box plot of ages
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['age'])
    plt.title('Age Distribution Box Plot', fontsize=16)
    plt.ylabel('Age', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_boxplot.png'), dpi=300)
    plt.close()
    
    print("Successfully created all visualizations.")
except Exception as e:
    print(f"Warning: Could not create visualizations due to: {e}")
    print("Continuing with just the data tables...")

print("\nAnalysis complete. All files saved to:", output_dir)