import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from itertools import combinations

# Configuration - use the same as your original code
BASE_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features"
OUTPUT_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/subject_paired_ttest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_TYPES = ['fimo', 'deep', 'rp', 'reg']
PCA_COMPONENT = "0.95"
CFS_THRESHOLD = "0.7"
PCA_FEATURES = "120"  # Adjust this if needed for different datatype PCA components
CFS_FEATURES = "200"
FOLDS = range(1, 6)  # Folds 1-5

# Define PCA features per data type - you might need to adjust these values
PCA_FEATURES_BY_TYPE = {
    'fimo': '120',
    'deep': '120',
    'rp': '3',  # Based on your example
    'reg': '120'
}

def parse_classification_report(file_path):
    """
    Parse a classification report text file to extract subject-level metrics AND subject-wise predictions
    """
    metrics = {}
    subject_predictions = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Initialize flags to track which section we're in
        in_subject_section = False
        in_chunk_section = False
        in_subject_predictions = False
        subject_info_found = False
        
        # Try to find subject predictions by looking for patterns
        for i, line in enumerate(lines):
            # Check if this could be a subject prediction line
            # Look for lines with a pattern like: "subject_id 0 1" or "patient_123 1 0"
            # This would likely be ID, actual label, predicted label
            if not in_subject_predictions and i > 10:  # Skip header lines
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Check if the last two items look like 0/1 or class labels
                    try:
                        val1 = int(parts[-2])
                        val2 = int(parts[-1])
                        if all(x in [0, 1] for x in [val1, val2]):
                            # This might be a predictions line
                            in_subject_predictions = True
                    except (ValueError, IndexError):
                        pass
            
            # Determine which section we're in based on headers
            if "SUBJECT-LEVEL METRICS" in line or "Subject-level metrics" in line:
                in_subject_section = True
                in_chunk_section = False
                in_subject_predictions = False
                continue
            elif "CHUNK-LEVEL METRICS" in line or "Chunk-level metrics" in line:
                in_subject_section = False
                in_chunk_section = True
                in_subject_predictions = False
                continue
            elif "SUBJECT-WISE PREDICTIONS" in line or "Subject-wise predictions" in line or "SUBJECT PREDICTIONS" in line:
                in_subject_section = False
                in_chunk_section = False
                in_subject_predictions = True
                continue
            
            # Extract metrics based on the current section
            if in_subject_section:
                if 'Test Accuracy:' in line or 'Accuracy:' in line:
                    try:
                        metrics['accuracy'] = float(line.split(':')[1].strip())
                        subject_info_found = True
                    except (ValueError, IndexError):
                        pass
                elif 'F1 Score:' in line or 'F1:' in line:
                    try:
                        metrics['f1'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Precision:' in line:
                    try:
                        metrics['precision'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Recall:' in line:
                    try:
                        metrics['recall'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'AUC:' in line:
                    try:
                        auc_value = line.split(':')[1].strip()
                        metrics['auc'] = float(auc_value) if auc_value != 'N/A' else np.nan
                    except (ValueError, IndexError):
                        pass
                elif 'Number of test subjects:' in line:
                    try:
                        metrics['num_subjects'] = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Number of positive subjects:' in line:
                    try:
                        metrics['num_positive'] = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Number of negative subjects:' in line:
                    try:
                        metrics['num_negative'] = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
            
            # Extract subject-wise predictions if in that section
            elif in_subject_predictions:
                if line.strip() and not line.startswith('---') and not any(x in line.lower() for x in ["subject", "actual", "predicted"]):
                    parts = line.strip().split()
                    if len(parts) >= 3:  # Ensure we have subject ID, actual, and predicted
                        try:
                            subject_id = parts[0]
                            actual = int(parts[-2])
                            predicted = int(parts[-1])
                            subject_predictions[subject_id] = {
                                'actual': actual,
                                'predicted': predicted,
                                'correct': actual == predicted
                            }
                        except (ValueError, IndexError):
                            pass  # Skip lines that don't match our expected format
            
            # Fall back to chunk-level metrics if no subject section is found
            elif not subject_info_found and not in_subject_section and not in_chunk_section:
                if 'Test Accuracy:' in line or 'Accuracy:' in line:
                    try:
                        metrics['accuracy'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'F1 Score:' in line or 'F1:' in line:
                    try:
                        metrics['f1'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Precision:' in line:
                    try:
                        metrics['precision'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Recall:' in line:
                    try:
                        metrics['recall'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'AUC:' in line:
                    try:
                        auc_value = line.split(':')[1].strip()
                        metrics['auc'] = float(auc_value) if auc_value != 'N/A' else np.nan
                    except (ValueError, IndexError):
                        pass
        
        # If subject predictions were found, print information
        if subject_predictions:
            print(f"Found predictions for {len(subject_predictions)} subjects in {file_path}")
        
        # If no metrics were found, print a warning
        if not metrics:
            print(f"Warning: No metrics found in {file_path}")
            
    except Exception as e:
        print(f"Error parsing file {file_path}: {str(e)}")
        return {}, {}
    
    return metrics, subject_predictions

def collect_model_results(model_type='RF', feature_selection='CFS'):
    """
    Collect results from all data types for a specific model and feature selection method
    """
    results = []
    subject_results = {}  # Dictionary to store subject-level predictions by fold
    
    for data_type in DATA_TYPES:
        # Get correct PCA features for this data type
        if feature_selection == 'PCA':
            pca_features = PCA_FEATURES_BY_TYPE.get(data_type, PCA_FEATURES)
        else:
            pca_features = PCA_FEATURES  # Not used but keeping for consistency
            
        for fold in FOLDS:
            # Construct file paths based on model type and feature selection
            if model_type == 'RF':
                if feature_selection == 'CFS':
                    file_path = os.path.join(
                        BASE_DIR, 
                        data_type, 
                        'CFS', 
                        f'fold{fold}', 
                        f'threshold_{CFS_THRESHOLD}', 
                        f'classification_report_{CFS_FEATURES}_features_SMOTE_GRIDSEARCHFalse.txt'
                    )
                else:  # PCA
                    file_path = os.path.join(
                        BASE_DIR, 
                        data_type, 
                        f'PCA_{PCA_COMPONENT}', 
                        f'fold{fold}', 
                        f'features_{pca_features}', 
                        f'classification_report_{pca_features}_features_SMOTE_GRIDSEARCHFalse.txt'
                    )
                    # Check for the PCA prefix version of the file
                    if not os.path.exists(file_path):
                        file_path = os.path.join(
                            BASE_DIR, 
                            data_type, 
                            f'PCA_{PCA_COMPONENT}', 
                            f'fold{fold}', 
                            f'features_{pca_features}', 
                            f'PCA_classification_report_{pca_features}_features_SMOTE_GRIDSEARCHFalse.txt'
                        )
            else:  # XGBoost
                if feature_selection == 'CFS':
                    file_path = os.path.join(
                        BASE_DIR, 
                        data_type, 
                        'xgboost',
                        'CFS', 
                        f'fold{fold}', 
                        f'threshold_{CFS_THRESHOLD}', 
                        f'classification_report_CFS_{CFS_FEATURES}_features_SMOTE_GRIDSEARCHFalse.txt'
                    )
                else:  # PCA
                    file_path = os.path.join(
                        BASE_DIR, 
                        data_type, 
                        'xgboost',
                        f'PCA_{PCA_COMPONENT}', 
                        f'fold{fold}', 
                        f'features_{pca_features}', 
                        f'classification_report_{pca_features}_features_SMOTE_GRIDSEARCHFalse.txt'
                    )
                    # Check for the PCA prefix version of the file
                    if not os.path.exists(file_path):
                        file_path = os.path.join(
                            BASE_DIR, 
                            data_type, 
                            'xgboost',
                            f'PCA_{PCA_COMPONENT}', 
                            f'fold{fold}', 
                            f'features_{pca_features}', 
                            f'PCA_classification_report_{pca_features}_features_SMOTE_GRIDSEARCHFalse.txt'
                        )
            
            # Process the file if it exists
            if os.path.exists(file_path):
                print(f"Processing: {file_path}")
                # Parse the classification report to extract metrics
                metrics, subject_preds = parse_classification_report(file_path)
                
                # Add data type and fold information
                metrics['data_type'] = data_type
                metrics['fold'] = fold
                metrics['model_type'] = model_type
                metrics['feature_selection'] = feature_selection
                metrics['model'] = f"{model_type}_{feature_selection}"
                metrics['file_path'] = file_path  # Store the file path for debugging
                
                # Store subject-level predictions
                if subject_preds:
                    config_key = f"{data_type}_{model_type}_{feature_selection}"
                    if config_key not in subject_results:
                        subject_results[config_key] = {}
                    
                    subject_results[config_key][fold] = subject_preds
                
                results.append(metrics)
            else:
                print(f"Warning: File not found: {file_path}")
    
    df = pd.DataFrame(results)
    
    # Print summary of collected data
    print(f"\nCollected {len(df)} results for {model_type}_{feature_selection}")
    if not df.empty:
        # Check how many files had subject metrics (if that column exists)
        if 'num_subjects' in df.columns:
            subject_metrics = df['num_subjects'].notna().sum()
            print(f"Files with subject metrics: {subject_metrics}/{len(df)}")
        else:
            print("Column 'num_subjects' not found in the data")
        
        # Calculate mean metrics by data type for columns that exist
        metrics_cols = [col for col in ['accuracy', 'f1', 'precision', 'recall', 'auc'] if col in df.columns]
        if metrics_cols:
            summary = df.groupby('data_type')[metrics_cols].mean().round(3)
            print("\nAverage subject-level metrics by data type:")
            print(summary)
        else:
            print("No metric columns found in the data")
    
    return df, subject_results

def prepare_subject_level_data(all_subject_results):
    """
    Prepare data for subject-level paired analysis
    """
    # Reorganize subject-level data for paired analysis
    subject_paired_data = {}
    
    # Identify all unique subjects across all configurations
    all_subjects = set()
    for config, fold_data in all_subject_results.items():
        for fold, subjects in fold_data.items():
            all_subjects.update(subjects.keys())
    
    print(f"Found {len(all_subjects)} unique subjects across all configurations")
    
    # For each subject, collect the outcomes from different configurations
    for subject in all_subjects:
        subject_paired_data[subject] = {}
        
        for config, fold_data in all_subject_results.items():
            for fold, subjects in fold_data.items():
                if subject in subjects:
                    # Extract data type and model info from config
                    data_type, model_type, feature_selection = config.split('_', 2)
                    
                    # Store the prediction result for this subject under this configuration
                    if data_type not in subject_paired_data[subject]:
                        subject_paired_data[subject][data_type] = {}
                    
                    model_key = f"{model_type}_{feature_selection}"
                    if model_key not in subject_paired_data[subject][data_type]:
                        subject_paired_data[subject][data_type][model_key] = {}
                    
                    subject_paired_data[subject][data_type][model_key][fold] = subjects[subject]
    
    return subject_paired_data

def perform_subject_paired_ttests(subject_paired_data):
    """
    Perform truly paired t-tests at the subject level
    """
    print("\n===== SUBJECT-LEVEL PAIRED T-TESTS =====")
    
    results = []
    
    # 1. Compare data types (for each model, using the same subjects)
    print("\nComparing Data Types (Paired Subject Analysis):")
    
    # Get all subjects with predictions across multiple data types
    for subject, data_types in subject_paired_data.items():
        if len(data_types) < 2:
            continue  # Skip subjects with only one data type
            
        # Get all pairs of data types for this subject
        for (type1, type1_models), (type2, type2_models) in combinations(data_types.items(), 2):
            # Find common models between these data types
            common_models = set(type1_models.keys()) & set(type2_models.keys())
            
            for model in common_models:
                # Find common folds
                common_folds = set(type1_models[model].keys()) & set(type2_models[model].keys())
                
                # Check the correctness for each fold
                for fold in common_folds:
                    type1_correct = 1 if type1_models[model][fold]['correct'] else 0
                    type2_correct = 1 if type2_models[model][fold]['correct'] else 0
                    
                    # Store this paired observation
                    results.append({
                        'subject': subject,
                        'comparison': 'data_type',
                        'group1': type1,
                        'group2': type2,
                        'model': model,
                        'fold': fold,
                        'group1_correct': type1_correct,
                        'group2_correct': type2_correct,
                        'difference': type1_correct - type2_correct
                    })
    
    # Convert to DataFrame
    paired_df = pd.DataFrame(results)
    
    # Perform paired t-tests on the differences
    if not paired_df.empty:
        # Group the data by the comparison parameters
        grouped = paired_df.groupby(['comparison', 'group1', 'group2', 'model'])
        
        # Perform t-test for each group
        ttest_results = []
        
        for name, group in grouped:
            # Get the differences for this comparison
            differences = group['difference']
            
            # Perform a 1-sample t-test on the differences (paired test)
            t_stat, p_value = stats.ttest_1samp(differences, 0)
            
            # Calculate effect size (Cohen's d for paired samples)
            effect_size = differences.mean() / (differences.std() if differences.std() > 0 else 1)
            
            result = {
                'comparison': name[0],
                'group1': name[1],
                'group2': name[2],
                'model': name[3],
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': effect_size,
                'mean_diff': differences.mean(),
                'std_diff': differences.std(),
                'n_subjects': len(differences)
            }
            
            ttest_results.append(result)
            
            # Print the result
            sig_indicator = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"Data Type: {name[1]} vs {name[2]} (Model: {name[3]}): t={t_stat:.4f}, p={p_value:.4f} {sig_indicator}, d={effect_size:.4f}, mean_diff={differences.mean():.4f}, n={len(differences)}")
        
        # Convert to DataFrame and save
        ttest_results_df = pd.DataFrame(ttest_results)
        ttest_results_df.to_csv(os.path.join(OUTPUT_DIR, 'subject_paired_ttest_results.csv'), index=False)
        
        # Create visualizations
        create_subject_paired_ttest_visualizations(ttest_results_df, paired_df)
        
        return ttest_results_df, paired_df
    else:
        print("No paired subject data available for analysis")
        return None, None

def create_subject_paired_ttest_visualizations(ttest_df, paired_df):
    """
    Create visualizations for subject-level paired t-tests
    """
    # 1. Create bar chart of mean differences with error bars
    if not ttest_df.empty:
        # Create a group variable for plotting
        ttest_df['comparison_group'] = ttest_df['group1'] + ' vs ' + ttest_df['group2'] + ' (' + ttest_df['model'] + ')'
        
        # Sort by absolute mean difference
        ttest_df = ttest_df.assign(abs_mean_diff=ttest_df['mean_diff'].abs()).sort_values('abs_mean_diff', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            x='comparison_group', 
            y='mean_diff', 
            data=ttest_df,
            capsize=0.2
        )
        
        # Add error bars manually
        for i, row in ttest_df.iterrows():
            ax.errorbar(
                i, row['mean_diff'], 
                yerr=row['std_diff'] / np.sqrt(row['n_subjects']),  # Standard error
                fmt='none', color='black', capsize=5
            )
        
        # Add significance asterisks
        for i, row in ttest_df.iterrows():
            sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
            if sig:
                ax.text(i, row['mean_diff'] + (0.05 if row['mean_diff'] >= 0 else -0.1), 
                       sig, ha='center', fontsize=12)
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='--')
        
        plt.title('Subject-Level Paired Comparisons - Mean Differences', fontsize=14)
        plt.ylabel('Mean Difference in Correct Predictions', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(OUTPUT_DIR, 'subject_paired_mean_differences.png'), dpi=300)
        plt.close()
    
    # 2. Create heatmap of p-values
    if not ttest_df.empty:
        # Get unique models
        models = ttest_df['model'].unique()
        
        for model in models:
            # Filter for this model
            model_df = ttest_df[ttest_df['model'] == model]
            
            # Get unique data types for this model
            data_types = sorted(set(model_df['group1']) | set(model_df['group2']))
            
            # Create a matrix of p-values
            p_value_matrix = np.ones((len(data_types), len(data_types)))
            
            # Fill in the p-values
            for i, type1 in enumerate(data_types):
                for j, type2 in enumerate(data_types):
                    if i != j:  # Skip the diagonal
                        # Find the p-value for this comparison
                        p_values = model_df[
                            ((model_df['group1'] == type1) & (model_df['group2'] == type2)) |
                            ((model_df['group1'] == type2) & (model_df['group2'] == type1))
                        ]['p_value'].values
                        
                        if len(p_values) > 0:
                            p_value_matrix[i, j] = p_values[0]
            
            # Create the heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                p_value_matrix,
                annot=True,
                fmt='.4f',
                cmap='coolwarm_r',
                xticklabels=data_types,
                yticklabels=data_types,
                vmin=0,
                vmax=0.1,
                cbar_kws={'label': 'p-value'}
            )
            
            plt.title(f'Subject-Level Paired P-values ({model})', fontsize=14)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(OUTPUT_DIR, f'subject_paired_pvalues_{model}.png'), dpi=300)
            plt.close()
    
    # 3. Create violin plots of the paired differences
    if not paired_df.empty:
        # Create a composite group variable
        paired_df['comparison_group'] = paired_df['group1'] + ' vs ' + paired_df['group2'] + ' (' + paired_df['model'] + ')'
        
        # Sort the data
        group_order = paired_df.groupby('comparison_group')['difference'].mean().abs().sort_values(ascending=False).index
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = sns.violinplot(
            x='comparison_group',
            y='difference',
            data=paired_df,
            order=group_order,
            inner='box'
        )
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='--')
        
        plt.title('Distribution of Subject-Level Paired Differences', fontsize=14)
        plt.ylabel('Difference in Correct Predictions (Group1 - Group2)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(OUTPUT_DIR, 'subject_paired_difference_distributions.png'), dpi=300)
        plt.close()

def cohen_d(x, y):
    """
    Calculate Cohen's d effect size between two groups
    
    Parameters:
    x, y : array-like
        The samples to compute the effect size between
        
    Returns:
    d : float
        Cohen's d effect size
    """
    nx = len(x)
    ny = len(y)
    
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate standard deviations
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    
    # Calculate Cohen's d
    d = (mean_x - mean_y) / pooled_std if pooled_std != 0 else 0
    
    return d

def main():
    # Initialize a dictionary to store all subject-level results
    all_subject_results = {}
    
    # Collect results for all model types
    print("Collecting RF-CFS results...")
    rf_cfs_results, rf_cfs_subject_results = collect_model_results('RF', 'CFS')
    all_subject_results.update(rf_cfs_subject_results)
    
    print("Collecting RF-PCA results...")
    rf_pca_results, rf_pca_subject_results = collect_model_results('RF', 'PCA')
    all_subject_results.update(rf_pca_subject_results)
    
    print("Collecting XGBoost-CFS results...")
    xgb_cfs_results, xgb_cfs_subject_results = collect_model_results('xgboost', 'CFS')
    all_subject_results.update(xgb_cfs_subject_results)
    
    print("Collecting XGBoost-PCA results...")
    xgb_pca_results, xgb_pca_subject_results = collect_model_results('xgboost', 'PCA')
    all_subject_results.update(xgb_pca_subject_results)
    
    # Combine all results
    all_results = pd.concat([rf_cfs_results, rf_pca_results, xgb_cfs_results, xgb_pca_results])
    
    # Check if we have subject-level prediction data
    if all_subject_results:
        print("\nPreparing subject-level paired data...")
        subject_paired_data = prepare_subject_level_data(all_subject_results)
        
        print("\nPerforming subject-level paired t-tests...")
        ttest_results, paired_data = perform_subject_paired_ttests(subject_paired_data)
        
        if ttest_results is not None:
            # Print summary of significant findings
            sig_results = ttest_results[ttest_results['significant']]
            print(f"\nFound {len(sig_results)} significant differences in subject-level paired comparisons")
            
            if not sig_results.empty:
                print("\nSignificant subject-level findings:")
                for _, row in sig_results.iterrows():
                    print(f"  {row['group1']} vs {row['group2']} ({row['model']}): p={row['p_value']:.4f}, mean_diff={row['mean_diff']:.4f}")
        
        print("\nCompleted subject-level paired analysis")
    else:
        print("\nNo subject-level prediction data available for paired analysis")
        print("Falling back to traditional (unpaired) statistical tests...")
        # You could call your original t-test function here if needed

if __name__ == "__main__":
    main()