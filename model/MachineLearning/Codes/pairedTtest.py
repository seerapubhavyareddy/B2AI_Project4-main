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
CFS_THRESHOLD = "0.7"
PCA_FEATURES = "120"
CFS_FEATURES = "200"
FOLDS = range(1, 6)  # Folds 1-5

# Use your existing functions for collecting and parsing data
def parse_classification_report(file_path):
    """
    Parse a classification report text file to extract metrics
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
            metrics['auc'] = float(auc_value) if auc_value != 'N/A' else np.nan
            
    return metrics

def collect_model_results(model_type='RF', feature_selection='CFS'):
    """
    Collect results from all data types for a specific model and feature selection method
    """
    results = []
    
    for data_type in DATA_TYPES:
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
                        f'features_{PCA_FEATURES}', 
                        f'classification_report_{PCA_FEATURES}_features_SMOTE_GRIDSEARCHFalse.txt'
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
                metrics['model_type'] = model_type
                metrics['feature_selection'] = feature_selection
                metrics['model'] = f"{model_type}_{feature_selection}"
                
                results.append(metrics)
            else:
                print(f"Warning: File not found: {file_path}")
    
    return pd.DataFrame(results)

def perform_paired_ttests(results_df):
    """
    Perform paired t-tests to compare:
    1. Different data types
    2. Different model types
    3. Different feature selection methods
    """
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    ttest_results = []
    
    print("\n===== PAIRED T-TESTS =====")
    
    # 1. Compare different data types
    print("\nComparing Data Types:")
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
                    'comparison': 'data_type',
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
                
                # Print result
                sig_indicator = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{type1} vs {type2}: t={t_stat:.4f}, p={p_value:.4f} {sig_indicator}, d={result['effect_size']:.4f}, diff={result['mean_diff']:.4f}")
            else:
                print(f"Insufficient data for {type1} vs {type2} comparison")
    
    # 2. Compare model types
    print("\nComparing Model Types:")
    model_types = results_df['model_type'].unique()
    
    for metric in metrics:
        print(f"\n{metric.upper()} metric:")
        
        # Get all pairwise combinations of model types
        for model1, model2 in combinations(model_types, 2):
            # Check metric values across all folds
            model1_values = results_df[results_df['model_type'] == model1][metric].dropna()
            model2_values = results_df[results_df['model_type'] == model2][metric].dropna()
            
            # Only perform t-test if we have sufficient data
            if len(model1_values) > 1 and len(model2_values) > 1:
                t_stat, p_value = stats.ttest_ind(model1_values, model2_values, equal_var=False)
                
                result = {
                    'comparison': 'model_type',
                    'group1': model1,
                    'group2': model2,
                    'metric': metric,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cohen_d(model1_values, model2_values),
                    'mean_diff': model1_values.mean() - model2_values.mean(),
                    'group1_mean': model1_values.mean(),
                    'group2_mean': model2_values.mean(),
                    'group1_std': model1_values.std(),
                    'group2_std': model2_values.std(),
                    'group1_n': len(model1_values),
                    'group2_n': len(model2_values)
                }
                
                ttest_results.append(result)
                
                # Print result
                sig_indicator = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{model1} vs {model2}: t={t_stat:.4f}, p={p_value:.4f} {sig_indicator}, d={result['effect_size']:.4f}, diff={result['mean_diff']:.4f}")
            else:
                print(f"Insufficient data for {model1} vs {model2} comparison")
    
    # 3. Compare feature selection methods
    print("\nComparing Feature Selection Methods:")
    feature_selections = results_df['feature_selection'].unique()
    
    for metric in metrics:
        print(f"\n{metric.upper()} metric:")
        
        # Get all pairwise combinations of feature selection methods
        for feature1, feature2 in combinations(feature_selections, 2):
            # Check metric values across all folds
            feature1_values = results_df[results_df['feature_selection'] == feature1][metric].dropna()
            feature2_values = results_df[results_df['feature_selection'] == feature2][metric].dropna()
            
            # Only perform t-test if we have sufficient data
            if len(feature1_values) > 1 and len(feature2_values) > 1:
                t_stat, p_value = stats.ttest_ind(feature1_values, feature2_values, equal_var=False)
                
                result = {
                    'comparison': 'feature_selection',
                    'group1': feature1,
                    'group2': feature2,
                    'metric': metric,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cohen_d(feature1_values, feature2_values),
                    'mean_diff': feature1_values.mean() - feature2_values.mean(),
                    'group1_mean': feature1_values.mean(),
                    'group2_mean': feature2_values.mean(),
                    'group1_std': feature1_values.std(),
                    'group2_std': feature2_values.std(),
                    'group1_n': len(feature1_values),
                    'group2_n': len(feature2_values)
                }
                
                ttest_results.append(result)
                
                # Print result
                sig_indicator = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{feature1} vs {feature2}: t={t_stat:.4f}, p={p_value:.4f} {sig_indicator}, d={result['effect_size']:.4f}, diff={result['mean_diff']:.4f}")
            else:
                print(f"Insufficient data for {feature1} vs {feature2} comparison")
    
    # 4. ADDITIONAL ANALYSIS: Compare data types within each model+feature combination
    print("\nComparing Data Types Within Each Model+Feature Combination:")
    
    # Get all model+feature combinations
    model_feature_combos = results_df.groupby(['model_type', 'feature_selection']).size().reset_index()[['model_type', 'feature_selection']]
    
    for _, combo in model_feature_combos.iterrows():
        model = combo['model_type']
        feature = combo['feature_selection']
        
        print(f"\nFor {model} with {feature}:")
        
        # Filter data for this model+feature combination
        combo_df = results_df[(results_df['model_type'] == model) & (results_df['feature_selection'] == feature)]
        
        for metric in metrics:
            print(f"  {metric.upper()} metric:")
            
            # Get all pairwise combinations of data types
            for type1, type2 in combinations(DATA_TYPES, 2):
                # Check metric values across all folds
                type1_values = combo_df[combo_df['data_type'] == type1][metric].dropna()
                type2_values = combo_df[combo_df['data_type'] == type2][metric].dropna()
                
                # Only perform t-test if we have sufficient data
                if len(type1_values) > 1 and len(type2_values) > 1:
                    t_stat, p_value = stats.ttest_ind(type1_values, type2_values, equal_var=False)
                    
                    result = {
                        'comparison': f'data_type_within_{model}_{feature}',
                        'group1': type1,
                        'group2': type2,
                        'metric': metric,
                        'model_type': model,
                        'feature_selection': feature,
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
                    
                    # Print result
                    sig_indicator = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"    {type1} vs {type2}: t={t_stat:.4f}, p={p_value:.4f} {sig_indicator}, d={result['effect_size']:.4f}, diff={result['mean_diff']:.4f}")
                else:
                    print(f"    Insufficient data for {type1} vs {type2} comparison")
    
    # Convert to DataFrame and save results
    ttest_df = pd.DataFrame(ttest_results)
    ttest_df.to_csv(os.path.join(OUTPUT_DIR, 'paired_ttest_results.csv'), index=False)
    
    # Create visualizations
    create_ttest_visualizations(ttest_df)
    
    return ttest_df

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

def create_ttest_visualizations(ttest_df):
    """
    Create visualizations of t-test results
    """
    # 1. Create heatmap of p-values for each comparison type
    comparison_types = ttest_df['comparison'].unique()
    metrics = ttest_df['metric'].unique()
    
    for comp_type in comparison_types:
        # Filter data for this comparison type
        comp_df = ttest_df[ttest_df['comparison'] == comp_type]
        
        if 'within' in comp_type:
            # For the within-group comparisons, create separate visualizations for each metric
            for metric in metrics:
                metric_df = comp_df[comp_df['metric'] == metric]
                
                if not metric_df.empty:
                    # Get the unique model_type and feature_selection values
                    model_type = metric_df['model_type'].iloc[0]
                    feature_selection = metric_df['feature_selection'].iloc[0]
                    
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
                    sns.heatmap(
                        heatmap_df, 
                        annot=True, 
                        cmap='coolwarm_r', 
                        vmin=0, 
                        vmax=0.1, 
                        linewidths=.5, 
                        fmt='.4f',
                        cbar_kws={'label': 'p-value'}
                    )
                    plt.title(f'P-values for {metric.upper()} - {model_type} with {feature_selection}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, f'pvalues_{comp_type}_{metric}_{model_type}_{feature_selection}.png'), dpi=300)
                    plt.close()
        else:
            # For main comparison types, create a single heatmap for all metrics
            plt.figure(figsize=(12, 8))
            
            # Get unique groups for this comparison type
            groups = sorted(set(comp_df['group1'].unique()) | set(comp_df['group2'].unique()))
            
            # Create subplots for each metric
            fig, axes = plt.subplots(
                nrows=len(metrics), 
                ncols=1, 
                figsize=(8, 4 * len(metrics)), 
                squeeze=False
            )
            
            for i, metric in enumerate(metrics):
                metric_df = comp_df[comp_df['metric'] == metric]
                
                # Create a matrix for the heatmap
                heatmap_data = []
                for group1 in groups:
                    row = []
                    for group2 in groups:
                        if group1 == group2:
                            row.append(1.0)  # Diagonal elements
                        else:
                            # Find the p-value for this comparison
                            p_value = metric_df[(
                                (metric_df['group1'] == group1) & (metric_df['group2'] == group2) |
                                (metric_df['group1'] == group2) & (metric_df['group2'] == group1)
                            )]['p_value'].values
                            
                            if len(p_value) > 0:
                                row.append(p_value[0])
                            else:
                                row.append(np.nan)
                    heatmap_data.append(row)
                
                # Create DataFrame for the heatmap
                heatmap_df = pd.DataFrame(heatmap_data, index=groups, columns=groups)
                
                # Create the heatmap
                sns.heatmap(
                    heatmap_df, 
                    annot=True, 
                    cmap='coolwarm_r', 
                    vmin=0, 
                    vmax=0.1, 
                    linewidths=.5, 
                    fmt='.4f',
                    cbar_kws={'label': 'p-value'},
                    ax=axes[i, 0]
                )
                axes[i, 0].set_title(f'{metric.upper()}')
            
            fig.suptitle(f'P-values for {comp_type.replace("_", " ").title()} Comparisons')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'pvalues_{comp_type}.png'), dpi=300)
            plt.close()
    
    # 2. Create bar plots showing effect sizes
    for comp_type in comparison_types:
        if 'within' not in comp_type:  # Skip the within-group comparisons for this visualization
            # Filter data for this comparison type
            comp_df = ttest_df[ttest_df['comparison'] == comp_type]
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create the plot
            sns.barplot(
                x='group1', 
                y='mean_diff',
                hue='group2',
                data=comp_df[comp_df['metric'] == 'accuracy']  # Use accuracy for this plot
            )
            
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title(f'Mean Differences in Accuracy for {comp_type.replace("_", " ").title()} Comparisons')
            plt.ylabel('Mean Difference')
            plt.xticks(rotation=45)
            plt.legend(title=comp_type.split('_')[-1].title() if '_' in comp_type else comp_type.title())
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'mean_diffs_{comp_type}.png'), dpi=300)
            plt.close()
    
    # 3. Create summary table of significant results
    sig_results = ttest_df[ttest_df['significant']]
    sig_summary = sig_results.groupby(['comparison', 'metric']).size().reset_index(name='num_significant')
    sig_summary_wide = sig_summary.pivot(index='comparison', columns='metric', values='num_significant').fillna(0).astype(int)
    
    # Create heatmap of significant results
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        sig_summary_wide, 
        annot=True, 
        cmap='YlGnBu', 
        linewidths=.5, 
        fmt='d',
        cbar_kws={'label': 'Number of Significant Results'}
    )
    plt.title('Number of Significant Results by Comparison Type and Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'significant_results_summary.png'), dpi=300)
    plt.close()

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
    
    # Check for missing values
    if all_results.isnull().any().any():
        print("Warning: Dataset contains missing values. Handling them...")
        # Drop rows with missing values
        all_results = all_results.dropna()
        print(f"After removing rows with missing values: {len(all_results)} rows remain")
    
    # Save combined results to the output directory
    all_results.to_csv(os.path.join(OUTPUT_DIR, 'all_model_results.csv'), index=False)
    print(f"Saved combined results with {len(all_results)} rows")
    
    # Perform paired t-tests
    print("\nPerforming paired t-tests...")
    ttest_results = perform_paired_ttests(all_results)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()