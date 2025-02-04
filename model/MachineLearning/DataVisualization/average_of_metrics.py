import os
import numpy as np
import re
import csv

# Function to parse the classification report and extract the values
def parse_classification_report(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extracting test accuracy, precision, recall, f1-score, and AUC using regular expressions
    accuracy = float(re.search(r'Test Accuracy: (\d+\.\d+)', content).group(1))
    f1_score = float(re.search(r'F1 Score: (\d+\.\d+)', content).group(1))
    precision = float(re.search(r'Precision: (\d+\.\d+)', content).group(1))
    recall = float(re.search(r'Recall: (\d+\.\d+)', content).group(1))
    auc = float(re.search(r'AUC: (\d+\.\d+)', content).group(1))
    
    # Returning the values as a tuple
    return accuracy, f1_score, precision, recall, auc

# Path to the directory containing the results files
base_path = '/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/rp/xgboost/CFS/'

# Thresholds to analyze
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

# Initialize lists to store results
results = []
accuracy_list = []
f1_score_list = []
precision_list = []
recall_list = []
auc_list = []

# Loop over the folds and thresholds to collect the data
for fold in range(1, 6):
    fold_results = [fold]  # Start with the fold number
    accuracy_fold = []
    f1_score_fold = []
    precision_fold = []
    recall_fold = []
    auc_fold = []

    for threshold in thresholds:
        # Construct the file path based on the fold and threshold
        file_path = os.path.join(base_path, f'fold{fold}/threshold_{threshold}/classification_report_CFS_200_features_SMOTE_GRIDSEARCHFalse.txt')

        # Check if the file exists
        if os.path.exists(file_path):
            # Parse the classification report
            accuracy, f1_score, precision, recall, auc = parse_classification_report(file_path)
            fold_results.append(accuracy)  # Add the accuracy to the results
            
            # Store individual metric results for averaging later
            accuracy_fold.append(accuracy)
            f1_score_fold.append(f1_score)
            precision_fold.append(precision)
            recall_fold.append(recall)
            auc_fold.append(auc)
        else:
            fold_results.append(None)  # If the file does not exist, append None

    # Add the fold's results to the main results list
    results.append(fold_results)

    # Append the metrics for average calculation
    accuracy_list.append(accuracy_fold)
    f1_score_list.append(f1_score_fold)
    precision_list.append(precision_fold)
    recall_list.append(recall_fold)
    auc_list.append(auc_fold)

# Calculate the averages for each threshold and each metric
avg_accuracy = np.nanmean(accuracy_list, axis=0)
avg_f1_score = np.nanmean(f1_score_list, axis=0)
avg_precision = np.nanmean(precision_list, axis=0)
avg_recall = np.nanmean(recall_list, axis=0)
avg_auc = np.nanmean(auc_list, axis=0)

# Append the average row to the results
results.append(['Avg'] + list(avg_accuracy))

# Define the file path to store the results
output_file = '/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/deep/results/cfs_xgboost_rp_classification_results_separated.csv'

# Write the results to a CSV file with separate tables for each metric
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)

    # Write Accuracy Table
    writer.writerow(["fold", "0.5", "0.6", "0.7", "0.8", "0.9", "avg_accuracy"])
    for fold_results, acc in zip(results[:-1], avg_accuracy):
        writer.writerow(fold_results + [acc])
    writer.writerow(['Avg'] + list(avg_accuracy))

    # Write F1 Score Table
    writer.writerow([])  # Add a blank line for separation
    writer.writerow(["fold", "0.5", "0.6", "0.7", "0.8", "0.9", "avg_f1_score"])
    for fold_results, f1 in zip(results[:-1], avg_f1_score):
        writer.writerow(fold_results + [f1])
    writer.writerow(['Avg'] + list(avg_f1_score))

    # Write Precision Table
    writer.writerow([])  # Add a blank line for separation
    writer.writerow(["fold", "0.5", "0.6", "0.7", "0.8", "0.9", "avg_precision"])
    for fold_results, prec in zip(results[:-1], avg_precision):
        writer.writerow(fold_results + [prec])
    writer.writerow(['Avg'] + list(avg_precision))

    # Write Recall Table
    writer.writerow([])  # Add a blank line for separation
    writer.writerow(["fold", "0.5", "0.6", "0.7", "0.8", "0.9", "avg_recall"])
    for fold_results, rec in zip(results[:-1], avg_recall):
        writer.writerow(fold_results + [rec])
    writer.writerow(['Avg'] + list(avg_recall))

    # Write AUC Table
    writer.writerow([])  # Add a blank line for separation
    writer.writerow(["fold", "0.5", "0.6", "0.7", "0.8", "0.9", "avg_auc"])
    for fold_results, auc in zip(results[:-1], avg_auc):
        writer.writerow(fold_results + [auc])
    writer.writerow(['Avg'] + list(avg_auc))

print(f"Results saved to {output_file}")
