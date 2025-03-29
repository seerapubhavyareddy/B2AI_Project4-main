import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from itertools import combinations

# Configuration - use the same as your original code
BASE_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features"
OUTPUT_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/paired_ttest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_TYPES = ['fimo', 'deep', 'rp', 'reg']
PCA_COMPONENT = "0.95"
PCA_FEATURES = "120"
FOLDS = range(1, 6)  # Folds 1-5

# Use your updated parse function for AUC extraction
def parse_classification_report(file_path):
    """
    Parse a classification report text file to extract metrics including AUC
    """
    metrics = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if 'Accuracy:' in line:
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

def collect_xgboost_pca_results():
    """
    Collect results for XGBoost-PCA across all data types
    """
    results = []
    
    for data_type in DATA_TYPES:
        for fold in FOLDS:
            # Construct file path for XGBoost-PCA
            file_path = os.path.join(
                BASE_DIR, 
                data_type, 
                'xgboost',
                f'PCA_{PCA_COMPONENT}', 
                f'fold{fold}', 
                f'features_{PCA_FEATURES}', 
                f'PCA_classification_report_{PCA_FEATURES}_features_SMOTE_GRIDSEARCHFalse.txt'
            )
            
            # Process the file if it exists
            if os.path.exists(file_path):
                # Parse the classification report to extract metrics
                metrics = parse_classification_report(file_path)
                
                # Add data type and fold information
                metrics['data_type'] = data_type
                metrics['fold'] = fold
                metrics['model_type'] = 'xgboost'
                metrics['feature_selection'] = 'PCA'
                metrics['model'] = 'xgboost_PCA'
                
                results.append(metrics)
            else:
                print(f"Warning: File not found: {file_path}")
    
    return pd.DataFrame(results)

def cohen_d(x, y):
    """
    Calculate Cohen's d effect size between two groups
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

def perform_focused_ttests(results_df):
    """
    Perform paired t-tests to compare data types within the XGBoost-PCA configuration
    """
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    ttest_results = []
    
    print("\n===== PAIRED T-TESTS FOR DATA TYPES WITHIN XGBOOST-PCA =====")
    
    for metric in metrics:
        print(f"\n{metric.upper()} metric:")
        
        # Get all pairwise combinations of data types
        for type1, type2 in combinations(DATA_TYPES, 2):
            # Check metric values across all folds
            type1_values = results_df[results_df['data_type'] == type1][metric].dropna()
            type2_values = results_df[results_df['data_type'] == type2][metric].dropna()
            
            # Only perform t-test if we have sufficient data
            if len(type1_values) > 1 and len(type2_values) > 1:
                t_stat, p_value = stats.ttest_ind(type1_values, type2_values, equal_var=False)
                
                result = {
                    'comparison': 'data_type_within_xgboost_PCA',
                    'group1': type1,
                    'group2': type2,
                    'metric': metric,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cohen_d(type1_values, type2_values),
                    'mean_diff': type1_values.mean() - type2_values.mean(),
                    'group1_mean': type1_values.mean(),
                    'group2_mean': type2_values.mean(),
                    'group1_std': type1_values.std(),
                    'group2_std': type2_values.std(),
                    'group1_n': len(type1_values),
                    'group2_n': len(type2_values)
                }
                
                ttest_results.append(result)
                
                # Print result with significance indicator
                sig_indicator = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{type1} vs {type2}: t={t_stat:.4f}, p={p_value:.4f} {sig_indicator}, d={result['effect_size']:.4f}, diff={result['mean_diff']:.4f}")
            else:
                print(f"Insufficient data for {type1} vs {type2} comparison")
    
    # Convert to DataFrame and save results
    ttest_df = pd.DataFrame(ttest_results)
    ttest_df.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_pca_data_type_ttest_results.csv'), index=False)
    
    # Create focused visualizations
    create_focused_visualizations(ttest_df)
    
    return ttest_df

def create_focused_visualizations(ttest_df):
    """
    Create visualizations of t-test results for data types within XGBoost-PCA
    """
    metrics = ttest_df['metric'].unique()
    
    # 1. Create a heatmap of p-values
    for metric in metrics:
        metric_df = ttest_df[ttest_df['metric'] == metric]
        
        if not metric_df.empty:
            # Create a matrix for the heatmap
            heatmap_data = []
            data_types = sorted(set(metric_df['group1'].unique()) | set(metric_df['group2'].unique()))
            
            for type1 in data_types:
                row = []
                for type2 in data_types:
                    if type1 == type2:
                        row.append(1.0)  # Diagonal elements
                    else:
                        # Find the p-value for this comparison
                        p_value = metric_df[(
                            (metric_df['group1'] == type1) & (metric_df['group2'] == type2) |
                            (metric_df['group1'] == type2) & (metric_df['group2'] == type1)
                        )]['p_value'].values
                        
                        if len(p_value) > 0:
                            row.append(p_value[0])
                        else:
                            row.append(np.nan)
                heatmap_data.append(row)
            
            # Create DataFrame for the heatmap
            heatmap_df = pd.DataFrame(heatmap_data, index=data_types, columns=data_types)
            
            # Create the heatmap
            plt.figure(figsize=(8, 6))
            mask = np.triu(np.ones_like(heatmap_df, dtype=bool))  # Create mask for lower triangle
            sns.heatmap(
                heatmap_df, 
                annot=True, 
                mask=mask,  # Only show lower triangle
                cmap='coolwarm_r', 
                vmin=0, 
                vmax=0.1, 
                linewidths=.5, 
                fmt='.4f',
                cbar_kws={'label': 'p-value'}
            )
            plt.title(f'P-values for {metric.upper()} - XGBoost with PCA')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'pvalues_data_type_xgboost_PCA_{metric}.png'), dpi=300)
            plt.close()
    
    # 2. Create bar plot comparing mean values across data types
    plt.figure(figsize=(12, 10))
    
    # Set up subplot grid for metrics
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        
        # Calculate mean values for each data type
        metric_means = {}
        metric_stds = {}
        
        for data_type in DATA_TYPES:
            data_rows = ttest_df[(ttest_df['group1'] == data_type) | (ttest_df['group2'] == data_type)]
            if data_type in data_rows['group1'].values:
                type_rows = data_rows[data_rows['group1'] == data_type]
                metric_means[data_type] = type_rows['group1_mean'].mean()
                metric_stds[data_type] = type_rows['group1_std'].mean()
            elif data_type in data_rows['group2'].values:
                type_rows = data_rows[data_rows['group2'] == data_type]
                metric_means[data_type] = type_rows['group2_mean'].mean()
                metric_stds[data_type] = type_rows['group2_std'].mean()
        
        # Create bar plot
        x = list(metric_means.keys())
        y = list(metric_means.values())
        yerr = list(metric_stds.values())
        
        bars = plt.bar(x, y, yerr=yerr, capsize=5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
            
        plt.title(f'{metric.capitalize()}')
        plt.ylim(0.65, 1.0)  # Set y-axis limits
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle('Performance Metrics by Breathing Type in XGBoost-PCA', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_pca_performance_by_data_type.png'), dpi=300)
    plt.close()
    
    # 3. Create a table visualization of significant results
    sig_results = ttest_df[ttest_df['significant']]
    
    # Count number of significant results by metric
    sig_count = sig_results.groupby('metric').size().reset_index(name='count')
    
    # Create a table to display significant differences
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    
    # Create the table with all significant results
    if not sig_results.empty:
        table_data = sig_results[['metric', 'group1', 'group2', 'p_value', 'effect_size', 'mean_diff']]
        table_data = table_data.sort_values(['metric', 'p_value'])
        
        # Format the data for display
        cell_text = []
        for _, row in table_data.iterrows():
            sig_stars = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*'
            cell_text.append([
                row['metric'].upper(),
                f"{row['group1']} vs {row['group2']}",
                f"{row['p_value']:.4f} {sig_stars}",
                f"{row['effect_size']:.4f}",
                f"{row['mean_diff']:.4f}"
            ])
        
        table = plt.table(
            cellText=cell_text,
            colLabels=['Metric', 'Comparison', 'p-value', "Cohen's d", 'Mean Diff'],
            loc='center',
            cellLoc='center'
        )
        
        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
    plt.title('Significant Differences Between Data Types in XGBoost-PCA')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_pca_significant_differences.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Collect XGBoost-PCA results
    print("Collecting XGBoost-PCA results...")
    xgb_pca_results = collect_xgboost_pca_results()
    
    # Check for missing values
    if xgb_pca_results.isnull().any().any():
        print("Warning: Dataset contains missing values. Handling them...")
        
        # Print which metrics have missing values
        for col in xgb_pca_results.columns:
            if xgb_pca_results[col].isnull().any():
                print(f"Column {col} has {xgb_pca_results[col].isnull().sum()} missing values")
        
        # Keep track of sample sizes before dropping
        before_counts = xgb_pca_results.groupby('data_type').size()
        print("Sample sizes before handling missing values:")
        print(before_counts)
        
        # Drop rows with missing values
        xgb_pca_results = xgb_pca_results.dropna()
        
        # Report sample sizes after dropping
        after_counts = xgb_pca_results.groupby('data_type').size()
        print("Sample sizes after handling missing values:")
        print(after_counts)
    
    # Save results to output directory
    xgb_pca_results.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_pca_results.csv'), index=False)
    print(f"Saved XGBoost-PCA results with {len(xgb_pca_results)} rows")
    
    # Calculate summary statistics for each data type
    summary = xgb_pca_results.groupby('data_type').agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'auc': ['mean', 'std']
    })
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Print summary
    print("\nSummary of XGBoost-PCA results by data type:")
    print(summary)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_pca_summary.csv'), index=False)
    
    # Perform paired t-tests
    print("\nPerforming paired t-tests...")
    ttest_results = perform_focused_ttests(xgb_pca_results)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()