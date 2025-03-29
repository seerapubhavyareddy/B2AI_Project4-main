import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import os
import glob
import scipy.stats

# Configuration
BASE_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features"
OUTPUT_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/anova_results_dataTypes"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

DATA_TYPES = ['fimo', 'deep', 'rp', 'reg']
PCA_COMPONENT = "0.95"
CFS_THRESHOLD = "0.7"
PCA_FEATURES = "120"
CFS_FEATURES = "200"
FOLDS = range(1, 6)  # Folds 1-5

# Define PCA features per data type
PCA_FEATURES_BY_TYPE = {
    'fimo': '120',
    'deep': '120',
    'rp': '3',
    'reg': '120'
}

def collect_model_results(model_type='RF', feature_selection='CFS'):
    """
    Collect results from all data types for a specific model and feature selection method
    
    Parameters:
    model_type: 'RF' or 'xgboost'
    feature_selection: 'CFS' or 'PCA'
    
    Returns:
    DataFrame with results for all data types
    """
    results = []
    
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
                    # Try with and without the PCA_ prefix in the filename
                    file_path = os.path.join(
                        BASE_DIR, 
                        data_type, 
                        f'PCA_{PCA_COMPONENT}', 
                        f'fold{fold}', 
                        f'features_{pca_features}', 
                        f'classification_report_{pca_features}_features_SMOTE_GRIDSEARCHFalse.txt'
                    )
                    
                    # If file doesn't exist, try with PCA_ prefix
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
                    # Try with and without the PCA_ prefix in the filename
                    file_path = os.path.join(
                        BASE_DIR, 
                        data_type, 
                        'xgboost',
                        f'PCA_{PCA_COMPONENT}', 
                        f'fold{fold}', 
                        f'features_{pca_features}', 
                        f'classification_report_{pca_features}_features_SMOTE_GRIDSEARCHFalse.txt'
                    )
                    
                    # If file doesn't exist, try with PCA_ prefix
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
                # Parse the classification report to extract metrics
                metrics = parse_classification_report(file_path)
                
                # Add data type and fold information
                metrics['data_type'] = data_type
                metrics['fold'] = fold
                metrics['model'] = f"{model_type}_{feature_selection}"
                
                results.append(metrics)
            else:
                print(f"Warning: File not found: {file_path}")
    
    return pd.DataFrame(results)

def parse_classification_report(file_path):
    """
    Parse a classification report text file to extract subject-level metrics
    """
    metrics = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize flags to track which section we're in
    in_subject_section = False
    in_chunk_section = False
    subject_info_found = False
    
    for line in lines:
        # Determine which section we're in
        if "SUBJECT-LEVEL METRICS" in line:
            in_subject_section = True
            in_chunk_section = False
            continue
        elif "CHUNK-LEVEL METRICS" in line:
            in_subject_section = False
            in_chunk_section = True
            continue
        
        # Extract metrics based on the current section
        if in_subject_section:
            if 'Test Accuracy:' in line:
                metrics['accuracy'] = float(line.split(':')[1].strip())
                subject_info_found = True
            elif 'F1 Score:' in line:
                metrics['f1'] = float(line.split(':')[1].strip())
            elif 'Precision:' in line:
                metrics['precision'] = float(line.split(':')[1].strip())
            elif 'Recall:' in line:
                metrics['recall'] = float(line.split(':')[1].strip())
            elif 'AUC:' in line:
                auc_value = line.split(':')[1].strip()
                # Handle 'N/A' values properly
                if auc_value != 'N/A' and auc_value:
                    try:
                        metrics['auc'] = float(auc_value)
                    except ValueError:
                        print(f"Warning: Could not convert AUC value '{auc_value}' to float in {file_path}")
                        metrics['auc'] = np.nan
                else:
                    metrics['auc'] = np.nan
            elif 'Number of test subjects:' in line:
                metrics['num_subjects'] = int(line.split(':')[1].strip())
            elif 'Number of positive subjects:' in line:
                metrics['num_positive'] = int(line.split(':')[1].strip())
            elif 'Number of negative subjects:' in line:
                metrics['num_negative'] = int(line.split(':')[1].strip())
        
        # Fall back to chunk-level metrics if no subject section is found
        elif not subject_info_found and not in_subject_section and not in_chunk_section:
            if 'Test Accuracy:' in line or 'Accuracy:' in line:
                metrics['accuracy'] = float(line.split(':')[1].strip())
            elif 'F1 Score:' in line:
                metrics['f1'] = float(line.split(':')[1].strip())
            elif 'Precision:' in line:
                metrics['precision'] = float(line.split(':')[1].strip())
            elif 'Recall:' in line:
                metrics['recall'] = float(line.split(':')[1].strip())
            elif 'AUC:' in line:
                auc_value = line.split(':')[1].strip()
                # Handle 'N/A' values properly
                if auc_value != 'N/A' and auc_value:
                    try:
                        metrics['auc'] = float(auc_value)
                    except ValueError:
                        print(f"Warning: Could not convert AUC value '{auc_value}' to float in {file_path}")
                        metrics['auc'] = np.nan
                else:
                    metrics['auc'] = np.nan
    
    # Check if we got all metrics
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
        if metric not in metrics:
            print(f"Warning: {metric} not found in {file_path}")
            metrics[metric] = np.nan
            
    return metrics

def analyze_performance_by_breathing_type(results_df):
    """
    Analyze and visualize performance differences across breathing types, including AUC
    """
    # Group by model and data_type to get average performance
    grouped = results_df.groupby(['model', 'data_type']).agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'auc': ['mean', 'std']  # Add AUC metrics
    })
    
    # Flatten the column hierarchy for easier access
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Create performance comparison plots
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']  # Add AUC to metrics
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        # Get unique models for legend
        models = grouped['model'].unique()
        bar_width = 0.2
        x_positions = np.arange(len(DATA_TYPES))
        
        for i, model in enumerate(models):
            model_data = grouped[grouped['model'] == model]
            
            # Create a mapping from data_type to position in the array
            data_type_to_idx = {data_type: idx for idx, data_type in enumerate(DATA_TYPES)}
            
            # Initialize arrays for plotting
            means = np.zeros(len(DATA_TYPES))
            stds = np.zeros(len(DATA_TYPES))
            
            # Fill in values for data types present in model_data
            for _, row in model_data.iterrows():
                data_type = row['data_type']
                data_type_idx = data_type_to_idx[data_type]
                means[data_type_idx] = row[f'{metric}_mean']
                stds[data_type_idx] = row[f'{metric}_std']
            
            # Create the bar positions
            positions = x_positions + (i - 1.5) * bar_width
            
            plt.bar(positions, means, width=bar_width, label=model, yerr=stds, capsize=5)
        
        plt.xlabel('Breathing Type')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Model Performance by Breathing Type - {metric.capitalize()}')
        plt.xticks(x_positions, DATA_TYPES)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'performance_by_breathing_type_{metric}.png'), dpi=300)
        plt.close()
        
    # Create a combined performance plot that now includes AUC
    plt.figure(figsize=(20, 15))  # Increased height to accommodate 5 metrics
    
    # Set up subplot grid for 5 metrics
    for idx, metric in enumerate(metrics):
        plt.subplot(3, 2, idx+1)  # Changed to 3x2 grid
        
        for i, model in enumerate(models):
            model_data = grouped[grouped['model'] == model]
            
            # Create a mapping from data_type to position in the array
            data_type_to_idx = {data_type: idx for idx, data_type in enumerate(DATA_TYPES)}
            
            # Initialize arrays for plotting
            means = np.zeros(len(DATA_TYPES))
            stds = np.zeros(len(DATA_TYPES))
            
            # Fill in values for data types present in model_data
            for _, row in model_data.iterrows():
                data_type = row['data_type']
                data_type_idx = data_type_to_idx[data_type]
                means[data_type_idx] = row[f'{metric}_mean']
                stds[data_type_idx] = row[f'{metric}_std']
            
            # Create the bar positions
            positions = x_positions + (i - 1.5) * bar_width
            
            plt.bar(positions, means, width=bar_width, label=model, yerr=stds, capsize=5)
        
        plt.xlabel('Breathing Type')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'{metric.capitalize()}')
        plt.xticks(x_positions, DATA_TYPES)
        if idx == 0:  # Only show legend on first subplot
            plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle('Model Performance by Breathing Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_performance_by_breathing_type.png'), dpi=300)
    plt.close()
    
    # Perform ANOVA test to check for statistically significant differences
    perform_statistical_tests(results_df)

def perform_statistical_tests(results_df):
    """
    Perform statistical tests to determine if differences between breathing types are significant,
    including AUC analysis
    """
    models = results_df['model'].unique()
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']  # Add AUC to metrics
    
    results = []
    anova_tables = []
    
    for model in models:
        model_data = results_df[results_df['model'] == model]
        
        for metric in metrics:
            # Collect data for each breathing type
            breathing_data = {}
            for data_type in DATA_TYPES:
                type_data = model_data[model_data['data_type'] == data_type][metric].values
                # Filter out NaN values for this analysis
                type_data = type_data[~np.isnan(type_data)]
                if len(type_data) > 0:
                    breathing_data[data_type] = type_data
            
            # Only perform ANOVA if we have at least two groups with data
            if len(breathing_data) >= 2:
                # Print sample sizes
                sample_sizes = {k: len(v) for k, v in breathing_data.items()}
                
                # Calculate ANOVA components
                groups = list(breathing_data.values())
                n_total = sum(len(group) for group in groups)
                k = len(groups)  # Number of groups
                
                # Calculate means for each group
                group_means = [np.mean(group) for group in groups]
                grand_mean = np.mean([val for group in groups for val in group])
                
                # Calculate Sum of Squares
                ss_between = sum(len(group) * (mean - grand_mean)**2 for group, mean in zip(groups, group_means))
                ss_within = sum(sum((val - mean)**2 for val in group) for group, mean in zip(groups, group_means))
                ss_total = ss_between + ss_within
                
                # Calculate degrees of freedom
                df_between = k - 1
                df_within = n_total - k
                df_total = n_total - 1
                
                # Calculate Mean Squares
                ms_between = ss_between / df_between
                ms_within = ss_within / df_within
                
                # Calculate F-statistic and p-value
                f_stat = ms_between / ms_within
                p_value = 1 - scipy.stats.f.cdf(f_stat, df_between, df_within)
                
                # Create ANOVA table row
                anova_table = {
                    'model': model,
                    'metric': metric,
                    'Source': ['Between Groups', 'Within Groups', 'Total'],
                    'SS': [ss_between, ss_within, ss_total],
                    'df': [df_between, df_within, df_total],
                    'MS': [ms_between, ms_within, np.nan],
                    'F': [f_stat, np.nan, np.nan],
                    'p': [p_value, np.nan, np.nan],
                    'significant': [p_value < 0.05, np.nan, np.nan]
                }
                
                anova_tables.append(pd.DataFrame(anova_table))
                
                # Add to simplified results for the heatmap
                result = {
                    'model': model,
                    'metric': metric,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'sample_sizes': str(sample_sizes),
                    'effect_size': ss_between / ss_total  # Adding eta-squared
                }
                
                results.append(result)
            else:
                print(f"Warning: Not enough groups for ANOVA - {model}, {metric}")
                
                result = {
                    'model': model,
                    'metric': metric,
                    'f_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'sample_sizes': str({k: len(v) for k, v in breathing_data.items()}),
                    'note': "Not enough groups for ANOVA",
                    'effect_size': np.nan
                }
                
                results.append(result)
    
    # Create a summary table
    stats_df = pd.DataFrame(results)
    
    # Combine all ANOVA tables
    full_anova_table = pd.concat(anova_tables)
    
    # Save full ANOVA tables to CSV
    full_anova_table.to_csv(os.path.join(OUTPUT_DIR, 'full_anova_tables.csv'), index=False)
    
    # Print results
    print("\n===== STATISTICAL ANALYSIS =====")
    for model in models:
        model_stats = stats_df[stats_df['model'] == model]
        print(f"\nStatistical analysis for {model}:")
        for _, row in model_stats.iterrows():
            if pd.isna(row['p_value']):
                print(f"- {row['metric'].capitalize()}: {row['note']}")
            else:
                significance = "statistically significant" if row['significant'] else "not statistically significant"
                # Check if we have the sample sizes for computing the degrees of freedom
                if "Not enough groups" not in row.get('note', ""):
                    # Parse the sample sizes dictionary string to get the total number of samples
                    sample_sizes_str = row['sample_sizes']
                    try:
                        # This is a bit of a hack to evaluate the string representation of the dictionary
                        sample_sizes_dict = eval(sample_sizes_str)
                        total_samples = sum(sample_sizes_dict.values())
                        num_groups = len(sample_sizes_dict)
                        df1 = num_groups - 1
                        df2 = total_samples - num_groups
                        print(f"- {row['metric'].capitalize()}: F({df1},{df2})={row['f_statistic']:.4f}, p={row['p_value']:.4f}, η²={row['effect_size']:.4f} ({significance})")
                    except:
                        print(f"- {row['metric'].capitalize()}: F-stat={row['f_statistic']:.4f}, p={row['p_value']:.4f}, η²={row['effect_size']:.4f} ({significance})")
                else:
                    print(f"- {row['metric'].capitalize()}: {row['note']}")
    
    # Save to file
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_tests_breathing_types.csv'), index=False)
    
    # Create table visualization of p-values
    plt.figure(figsize=(10, 8))
    pivot_stats = stats_df.pivot(index='model', columns='metric', values='p_value')
    sns.heatmap(pivot_stats, annot=True, cmap='coolwarm_r', vmin=0, vmax=0.1, 
                linewidths=.5, cbar_kws={'label': 'p-value'})
    plt.title('P-values from ANOVA Tests Across Models and Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'statistical_significance_heatmap.png'), dpi=300)
    plt.close()
    
    # Create effect size heatmap
    plt.figure(figsize=(10, 8))
    pivot_effect = stats_df.pivot(index='model', columns='metric', values='effect_size')
    sns.heatmap(pivot_effect, annot=True, cmap='viridis', vmin=0, vmax=1, 
                linewidths=.5, cbar_kws={'label': 'Effect Size (η²)'})
    plt.title('Effect Sizes (η²) from ANOVA Tests Across Models and Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'effect_size_heatmap.png'), dpi=300)
    plt.close()
    
    # Create a table for each model-metric combination
    for model in models:
        for metric in metrics:
            model_metric_data = full_anova_table[(full_anova_table['model'] == model) & 
                                               (full_anova_table['metric'] == metric)]
            
            if not model_metric_data.empty:
                # Create a formatted table
                table_data = model_metric_data[['Source', 'SS', 'df', 'MS', 'F', 'p']]
                
                # Formatting table
                plt.figure(figsize=(10, 2))
                plt.axis('off')
                
                # Create the table
                table = plt.table(
                    cellText=table_data[['Source', 'SS', 'df', 'MS', 'F', 'p']].round(4).values,
                    colLabels=['Source', 'SS', 'df', 'MS', 'F', 'p'],
                    cellLoc='center',
                    loc='center'
                )
                
                # Set table properties
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 1.5)
                
                # Add title
                plt.title(f'ANOVA Table: {model} - {metric.capitalize()}')
                
                # Save the table
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'anova_table_{model}_{metric}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # Create detailed results table
    detailed_df = results_df.groupby(['model', 'data_type']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'f1': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max'],
        'auc': ['mean', 'std', 'min', 'max']  # Add AUC metrics
    })
    
    # Flatten the column hierarchy for easier access
    detailed_df.columns = ['_'.join(col).strip() for col in detailed_df.columns.values]
    detailed_df = detailed_df.reset_index()
    
    # Save detailed results to CSV
    detailed_df.to_csv(os.path.join(OUTPUT_DIR, 'detailed_performance_by_breathing_type.csv'), index=False)

def main():
    # Collect results for all model types
    print("Collecting RF-CFS results...")
    rf_cfs_results = collect_model_results('RF', 'CFS')
    
    print("Collecting RF-PCA results...")
    rf_pca_results = collect_model_results('RF', 'PCA')
    
    print("Collecting XGBoost-CFS results...")
    xgb_cfs_results = collect_model_results('xgboost', 'CFS')
    
    print("Collecting XGBoost-PCA results...")
    xgb_pca_results = collect_model_results('xgboost', 'PCA')
    
    # Combine all results
    all_results = pd.concat([rf_cfs_results, rf_pca_results, xgb_cfs_results, xgb_pca_results])
    
    # Save combined results to the output directory
    all_results.to_csv(os.path.join(OUTPUT_DIR, 'all_model_results_by_breathing_type.csv'), index=False)
    print(f"Saved combined results with {len(all_results)} rows to {OUTPUT_DIR}")
    
    # Print summary
    summary = all_results.groupby(['model', 'data_type']).size().reset_index(name='count')
    print("\nSummary of collected results:")
    print(summary)
    
    # Also save the summary to the output directory
    summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_count_by_model_breathing_type.csv'), index=False)
    
    # Analyze performance differences
    print("\nAnalyzing performance by breathing type...")
    analyze_performance_by_breathing_type(all_results)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()