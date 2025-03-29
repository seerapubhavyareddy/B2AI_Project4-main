import os
import numpy as np
import re
import csv
import matplotlib.pyplot as plt
from pathlib import Path

def parse_classification_report(file_path):
    """Parse a classification report file to extract key metrics."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Extracting metrics using various patterns
        # Accuracy patterns
        accuracy_patterns = [
            r'Test Accuracy: (\d+\.\d+)',
            r'Accuracy: (\d+\.\d+)',
            r'accuracy: (\d+\.\d+)',
            r'ACC: (\d+\.\d+)'
        ]
        
        accuracy = None
        for pattern in accuracy_patterns:
            match = re.search(pattern, content)
            if match:
                accuracy = float(match.group(1))
                break
        
        # F1 Score patterns
        f1_patterns = [
            r'F1 Score: (\d+\.\d+)',
            r'f1-score: (\d+\.\d+)',
            r'F1: (\d+\.\d+)'
        ]
        
        f1_score = None
        for pattern in f1_patterns:
            match = re.search(pattern, content)
            if match:
                f1_score = float(match.group(1))
                break
        
        # Precision patterns
        precision_patterns = [
            r'Precision: (\d+\.\d+)',
            r'precision: (\d+\.\d+)'
        ]
        
        precision = None
        for pattern in precision_patterns:
            match = re.search(pattern, content)
            if match:
                precision = float(match.group(1))
                break
        
        # Recall patterns
        recall_patterns = [
            r'Recall: (\d+\.\d+)',
            r'recall: (\d+\.\d+)'
        ]
        
        recall = None
        for pattern in recall_patterns:
            match = re.search(pattern, content)
            if match:
                recall = float(match.group(1))
                break
        
        # AUC patterns
        auc_patterns = [
            r'AUC: (\d+\.\d+)',
            r'ROC AUC: (\d+\.\d+)',
            r'roc_auc: (\d+\.\d+)'
        ]
        
        auc = None
        for pattern in auc_patterns:
            match = re.search(pattern, content)
            if match:
                auc = float(match.group(1))
                break
        
        return accuracy, f1_score, precision, recall, auc
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None, None, None, None

def generate_reports(feature_type, algorithm):
    """Generate reports for a specific feature type and algorithm."""
    print(f"\n{'='*80}")
    print(f"Analyzing {feature_type.upper()} features with {algorithm.upper()}")
    print(f"{'='*80}")
    
    # Base paths
    base_dir = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features"
    
    # The exact path structure depends on the feature type and algorithm
    if feature_type == "deep":
        if algorithm == "xgboost":
            base_path = os.path.join(base_dir, "deep/xgboost/PCA_0.95")
        else:  # Assuming default case for other algorithms
            base_path = os.path.join(base_dir, f"deep/PCA_0.95")
    else:
        # For other feature types (reg, rp, fimo)
        if algorithm == "xgboost":
            base_path = os.path.join(base_dir, f"{feature_type}/xgboost/PCA_0.95")
        else:
            base_path = os.path.join(base_dir, f"{feature_type}/PCA_0.95")
    
    print(f"Base path: {base_path}")
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    # Features to analyze
    features = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    
    # Initialize data structures to store results
    all_results = {
        'accuracy': {feature: {fold: None for fold in range(1, 6)} for feature in features},
        'f1_score': {feature: {fold: None for fold in range(1, 6)} for feature in features},
        'precision': {feature: {fold: None for fold in range(1, 6)} for feature in features},
        'recall': {feature: {fold: None for fold in range(1, 6)} for feature in features},
        'auc': {feature: {fold: None for fold in range(1, 6)} for feature in features}
    }
    
    # Try different file patterns
    file_patterns = [
        lambda f, feat: f'fold{f}/features_{feat}/PCA_classification_report_{feat}_features_SMOTE_GRIDSEARCHFalse.txt',
        lambda f, feat: f'fold{f}/features_{feat}/classification_report_{feat}_features_SMOTE_GRIDSEARCHFalse.txt',
        lambda f, feat: f'fold{f}/features_{feat}/classification_report_{feat}_features.txt',
        lambda f, feat: f'fold{f}/features_{feat}/PCA_classification_report_{feat}.txt',
        lambda f, feat: f'fold{f}/features_{feat}/classification_report.txt'
    ]
    
    # Loop over the folds and features to collect the data
    found_files = 0
    for fold in range(1, 6):
        for feature in features:
            file_found = False
            
            for pattern_func in file_patterns:
                pattern = pattern_func(fold, feature)
                file_path = os.path.join(base_path, pattern)
                
                if os.path.exists(file_path):
                    print(f"Found file for fold {fold}, feature {feature}: {pattern}")
                    file_found = True
                    found_files += 1
                    
                    # Parse the classification report
                    accuracy, f1_score, precision, recall, auc = parse_classification_report(file_path)
                    
                    # Store the metrics
                    all_results['accuracy'][feature][fold] = accuracy
                    all_results['f1_score'][feature][fold] = f1_score
                    all_results['precision'][feature][fold] = precision
                    all_results['recall'][feature][fold] = recall
                    all_results['auc'][feature][fold] = auc
                    
                    break  # Exit after finding first matching file pattern
            
            if not file_found:
                print(f"No matching file found for fold {fold}, feature {feature}")
    
    print(f"Total files found: {found_files}")
    
    if found_files == 0:
        print(f"No files found for {feature_type} with {algorithm}. Skipping report generation.")
        return
    
    # Calculate averages for each feature and metric
    avg_metrics = {
        'accuracy': {feature: None for feature in features},
        'f1_score': {feature: None for feature in features},
        'precision': {feature: None for feature in features},
        'recall': {feature: None for feature in features},
        'auc': {feature: None for feature in features}
    }
    
    for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
        for feature in features:
            values = [all_results[metric][feature][fold] for fold in range(1, 6) if all_results[metric][feature][fold] is not None]
            avg_metrics[metric][feature] = np.mean(values) if values else None
    
    # Define the output directory
    output_dir = os.path.join(base_dir, f"{feature_type}/results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving results to: {output_dir}")
    
    # Create detailed tables for each metric with per-fold values
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
    for metric in metrics:
        # Create CSV file for this metric
        csv_file = os.path.join(output_dir, f'{feature_type}_{algorithm}_{metric}_by_fold.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header row
            writer.writerow(['Feature Count', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average'])
            
            # Write data rows
            for feature in sorted(features):
                row = [feature]
                
                # Add values for each fold
                for fold in range(1, 6):
                    value = all_results[metric][feature][fold]
                    row.append(f"{value:.4f}" if value is not None else "N/A")
                
                # Add average
                avg = avg_metrics[metric][feature]
                row.append(f"{avg:.4f}" if avg is not None else "N/A")
                
                writer.writerow(row)
        
        print(f"Created detailed table for {metric}: {csv_file}")
    
    # Create a summary CSV with average metrics
    summary_file = os.path.join(output_dir, f'{feature_type}_{algorithm}_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow(['Feature Count', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC'])
        
        # Write data rows
        for feature in sorted(features):
            row = [feature]
            
            for metric in metrics:
                avg = avg_metrics[metric][feature]
                row.append(f"{avg:.4f}" if avg is not None else "N/A")
            
            writer.writerow(row)
    
    print(f"Created summary table: {summary_file}")
    
    # Generate plots if we have enough data
    valid_features = []
    metric_values = {metric: [] for metric in metrics}
    
    for feature in sorted(features):
        if any(avg_metrics[metric][feature] is not None for metric in metrics):
            valid_features.append(feature)
            for metric in metrics:
                metric_values[metric].append(avg_metrics[metric][feature] if avg_metrics[metric][feature] is not None else None)
    
    if valid_features:
        # Create plots for each metric
        for metric in metrics:
            values = [avg_metrics[metric][feature] for feature in valid_features if avg_metrics[metric][feature] is not None]
            features_with_values = [feature for feature, value in zip(valid_features, [avg_metrics[metric][feature] for feature in valid_features]) if value is not None]
            
            if features_with_values:
                plt.figure(figsize=(10, 6))
                plt.plot(features_with_values, values, marker='o', linestyle='-', color='blue')
                plt.xlabel('Number of Features')
                plt.ylabel(f'Average {metric.replace("_", " ").title()}')
                plt.title(f'{feature_type.upper()} {algorithm.upper()}: Average {metric.replace("_", " ").title()} vs Number of Features')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(features_with_values)
                
                plot_file = os.path.join(output_dir, f'{feature_type}_{algorithm}_{metric}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Created plot for {metric}: {plot_file}")
    
        # Create a combined plot if we have enough data
        plt.figure(figsize=(12, 8))
        
        for metric in metrics:
            values = [avg_metrics[metric][feature] for feature in valid_features if avg_metrics[metric][feature] is not None]
            features_with_values = [feature for feature, value in zip(valid_features, [avg_metrics[metric][feature] for feature in valid_features]) if value is not None]
            
            if features_with_values:
                plt.plot(features_with_values, values, marker='o', linestyle='-', label=metric.replace("_", " ").title())
        
        plt.xlabel('Number of Features')
        plt.ylabel('Average Metric Value')
        plt.title(f'{feature_type.upper()} {algorithm.upper()}: Performance Metrics vs Number of Features')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        combined_plot_file = os.path.join(output_dir, f'{feature_type}_{algorithm}_all_metrics.png')
        plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created combined plot: {combined_plot_file}")
    else:
        print("Not enough valid data to create plots")
    
    # Create a markdown report
    markdown_content = f"""# {feature_type.upper()} {algorithm.upper()} Feature Analysis Report

## Summary of File Analysis
- Total files found and processed: {found_files}
- Features analyzed: {features}

"""
    
    # Add tables for each metric
    for metric in metrics:
        markdown_content += f"## {metric.replace('_', ' ').title()} Table\n"
        markdown_content += "| Feature Count | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |\n"
        markdown_content += "|---------------|--------|--------|--------|--------|--------|--------|\n"
        
        for feature in sorted(features):
            row = [str(feature)]
            
            # Add values for each fold
            for fold in range(1, 6):
                value = all_results[metric][feature][fold]
                row.append(f"{value:.4f}" if value is not None else "N/A")
            
            # Add average
            avg = avg_metrics[metric][feature]
            row.append(f"{avg:.4f}" if avg is not None else "N/A")
            
            markdown_content += f"| {' | '.join(row)} |\n"
        
        markdown_content += "\n"
    
    # Add conclusion
    markdown_content += f"""## Conclusion

This report provides a detailed analysis of {feature_type.upper()} {algorithm.upper()} model performance across different feature counts and cross-validation folds.
"""
    
    # Find best performing feature counts if we have data
    best_features = {}
    for metric in metrics:
        valid_values = [(feature, avg_metrics[metric][feature]) for feature in features if avg_metrics[metric][feature] is not None]
        if valid_values:
            best_feature, best_value = max(valid_values, key=lambda x: x[1])
            best_features[metric] = (best_feature, best_value)
    
    if best_features:
        markdown_content += "\n### Best Performing Feature Counts\n\n"
        for metric, (feature, value) in best_features.items():
            markdown_content += f"- Best {metric.replace('_', ' ').title()}: {value:.4f} (with {feature} features)\n"
    
    # Save the markdown report
    markdown_file = os.path.join(output_dir, f'{feature_type}_{algorithm}_detailed_report.md')
    with open(markdown_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Created markdown report: {markdown_file}")
    
    print(f"Analysis complete for {feature_type} with {algorithm}!")

def main():
    """Main function to analyze all feature types and algorithms."""
    # Feature types and algorithms to analyze
    feature_types = ["deep", "reg", "rp", "fimo"]
    algorithms = ["xgboost"]  # Add other algorithms if needed
    
    for feature_type in feature_types:
        for algorithm in algorithms:
            generate_reports(feature_type, algorithm)
    
    print("\nAll analyses complete!")

if __name__ == "__main__":
    main()