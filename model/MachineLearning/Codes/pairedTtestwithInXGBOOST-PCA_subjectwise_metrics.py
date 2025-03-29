import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import traceback

# Configuration
BASE_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features"
OUTPUT_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/xgboost_pca_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_TYPES = ['fimo', 'deep', 'rp', 'reg']
PCA_COMPONENT = "0.95"
PCA_FEATURES = "120"  # Base PCA features value
FOLDS = range(1, 6)  # Folds 1-5

# Define PCA features per data type
PCA_FEATURES_BY_TYPE = {
    'fimo': '120',
    'deep': '120',
    'rp': '3',  # Based on your example
    'reg': '120'
}

def parse_classification_report(file_path):
    """
    Parse a classification report text file to extract subject-level metrics
    """
    metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Initialize flags to track which section we're in
        in_subject_section = False
        in_chunk_section = False
        subject_info_found = False
        
        for line in lines:
            # Determine which section we're in based on headers
            if "SUBJECT-LEVEL METRICS" in line or "Subject-level metrics" in line:
                in_subject_section = True
                in_chunk_section = False
                continue
            elif "CHUNK-LEVEL METRICS" in line or "Chunk-level metrics" in line:
                in_subject_section = False
                in_chunk_section = True
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
        
        # If no metrics were found, print a warning
        if not metrics:
            print(f"Warning: No metrics found in {file_path}")
            
    except Exception as e:
        print(f"Error parsing file {file_path}: {str(e)}")
        return {}
    
    return metrics

def collect_xgboost_pca_results():
    """
    Collect results for XGBoost-PCA model from all data types
    """
    results = []
    
    for data_type in DATA_TYPES:
        # Get correct PCA features for this data type
        pca_features = PCA_FEATURES_BY_TYPE.get(data_type, PCA_FEATURES)
            
        for fold in FOLDS:
            # Construct file path for XGBoost-PCA
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
                metrics = parse_classification_report(file_path)
                
                # Add data type and fold information
                metrics['data_type'] = data_type
                metrics['fold'] = fold
                metrics['model'] = 'xgboost_PCA'
                metrics['file_path'] = file_path  # Store the file path for debugging
                
                results.append(metrics)
            else:
                print(f"Warning: File not found: {file_path}")
    
    df = pd.DataFrame(results)
    
    # Print summary of collected data
    print(f"\nCollected {len(df)} results for XGBoost-PCA model")
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
            print("\nAverage subject-level metrics by data type for XGBoost-PCA:")
            print(summary)
        else:
            print("No metric columns found in the data")
    
    return df

def perform_statistical_analysis(results_df):
    """
    Perform statistical analysis on fold-level results to compare data types
    """
    if results_df.empty:
        print("No results data available for analysis")
        return
    
    print("\n===== STATISTICAL ANALYSIS OF XGBOOST-PCA ACROSS DATA TYPES =====")
    
    # Check which metrics are available
    metrics = [col for col in ['accuracy', 'f1', 'precision', 'recall', 'auc'] if col in results_df.columns]
    
    # 1. One-way ANOVA to test if there are significant differences between data types
    anova_results = []
    
    for metric in metrics:
        # Prepare data for ANOVA
        data_by_type = [results_df[results_df['data_type'] == dt][metric].dropna().values for dt in DATA_TYPES]
        data_by_type = [d for d in data_by_type if len(d) > 0]  # Filter out empty arrays
        
        if len(data_by_type) >= 2:  # Need at least 2 groups for ANOVA
            try:
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*data_by_type)
                
                # Calculate effect size (eta-squared)
                # Sum of squares between groups / total sum of squares
                # Flatten all data
                all_data = np.concatenate(data_by_type)
                
                # Grand mean
                grand_mean = np.mean(all_data)
                
                # Sum of squares between groups
                ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in data_by_type)
                
                # Total sum of squares
                ss_total = sum((x - grand_mean)**2 for x in all_data)
                
                # Eta-squared
                eta_squared = ss_between / ss_total if ss_total != 0 else 0
                
                result = {
                    'metric': metric,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': eta_squared
                }
                
                anova_results.append(result)
                
                # Print result
                print(f"\nANOVA for {metric.upper()}:")
                print(f"F({len(data_by_type)-1}, {len(all_data)-len(data_by_type)}) = {f_stat:.4f}, p = {p_value:.4f}, η² = {eta_squared:.4f}")
                if p_value < 0.05:
                    print("Significant differences found between data types")
                else:
                    print("No significant differences between data types")
                
                # If significant, perform post-hoc tests
                if p_value < 0.05:
                    print("\nPost-hoc pairwise t-tests with Bonferroni correction:")
                    
                    # Prepare for post-hoc tests
                    data_type_values = {}
                    for dt in DATA_TYPES:
                        values = results_df[results_df['data_type'] == dt][metric].dropna().values
                        if len(values) > 0:
                            data_type_values[dt] = values
                    
                    # Perform all pairwise t-tests
                    posthoc_results = []
                    
                    for i, type1 in enumerate(data_type_values.keys()):
                        for j, type2 in enumerate(data_type_values.keys()):
                            if i < j:  # Only test each pair once
                                t_stat, p_val = stats.ttest_ind(
                                    data_type_values[type1], 
                                    data_type_values[type2],
                                    equal_var=False  # Welch's t-test (doesn't assume equal variances)
                                )
                                
                                # Calculate Cohen's d effect size
                                mean1 = np.mean(data_type_values[type1])
                                mean2 = np.mean(data_type_values[type2])
                                n1 = len(data_type_values[type1])
                                n2 = len(data_type_values[type2])
                                var1 = np.var(data_type_values[type1], ddof=1)
                                var2 = np.var(data_type_values[type2], ddof=1)
                                
                                # Pooled standard deviation
                                pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                                
                                # Cohen's d
                                cohens_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
                                
                                posthoc_results.append({
                                    'group1': type1,
                                    'group2': type2,
                                    'metric': metric,
                                    't_statistic': t_stat,
                                    'p_value': p_val,
                                    'bonferroni_p': p_val * len(data_type_values) * (len(data_type_values) - 1) / 2,
                                    'mean_diff': mean1 - mean2,
                                    'effect_size': cohens_d
                                })
                    
                    # Apply Bonferroni correction
                    for result in posthoc_results:
                        # Mark as significant if Bonferroni-corrected p-value < 0.05
                        result['significant'] = result['bonferroni_p'] < 0.05
                        
                        # Print result
                        sig_indicator = "***" if result['bonferroni_p'] < 0.001 else "**" if result['bonferroni_p'] < 0.01 else "*" if result['bonferroni_p'] < 0.05 else "ns"
                        print(f"{result['group1']} vs {result['group2']}: t = {result['t_statistic']:.4f}, p = {result['p_value']:.4f}, corrected p = {result['bonferroni_p']:.4f} {sig_indicator}, d = {result['effect_size']:.4f}, diff = {result['mean_diff']:.4f}")
                    
                    # Save post-hoc results
                    posthoc_df = pd.DataFrame(posthoc_results)
                    posthoc_df.to_csv(os.path.join(OUTPUT_DIR, f'xgboost_pca_posthoc_{metric}.csv'), index=False)
                    
                    # Create visualization of post-hoc results
                    create_posthoc_visualizations(posthoc_df, metric)
                    
            except Exception as e:
                print(f"Error performing ANOVA for {metric}: {str(e)}")
                traceback.print_exc()
        else:
            print(f"\nInsufficient data for ANOVA on {metric.upper()}")
    
    # Save ANOVA results
    if anova_results:
        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_pca_anova_results.csv'), index=False)
        
        # Create visualization of ANOVA results
        create_anova_visualizations(anova_df)
    
    # 2. Create a summary table and box plots
    create_summary_visualizations(results_df, metrics)

def create_posthoc_visualizations(posthoc_df, metric):
    """
    Create visualizations for post-hoc test results
    """
    if posthoc_df.empty:
        return
    
    # 1. Create heatmap of p-values
    plt.figure(figsize=(8, 6))
    
    # Get unique data types
    data_types = sorted(set(posthoc_df['group1']) | set(posthoc_df['group2']))
    
    # Create a matrix of p-values
    p_matrix = np.ones((len(data_types), len(data_types)))
    
    # Fill in the p-values (using Bonferroni-corrected p-values)
    for _, row in posthoc_df.iterrows():
        i = data_types.index(row['group1'])
        j = data_types.index(row['group2'])
        p_matrix[i, j] = row['bonferroni_p']
        p_matrix[j, i] = row['bonferroni_p']  # Make symmetric
    
    # Set diagonal to 1.0 (same group)
    np.fill_diagonal(p_matrix, 1.0)
    
    # Create the heatmap
    sns.heatmap(
        p_matrix,
        annot=True,
        fmt='.4f',
        cmap='coolwarm_r',
        xticklabels=data_types,
        yticklabels=data_types,
        vmin=0,
        vmax=0.1,
        cbar_kws={'label': 'Bonferroni-corrected p-value'}
    )
    
    plt.title(f'XGBoost-PCA: Post-hoc Test P-values for {metric.upper()}', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgboost_pca_posthoc_pvalues_{metric}.png'), dpi=300)
    plt.close()
    
    # 2. Create heatmap of effect sizes
    plt.figure(figsize=(8, 6))
    
    # Create a matrix of effect sizes
    d_matrix = np.zeros((len(data_types), len(data_types)))
    
    # Fill in the effect sizes
    for _, row in posthoc_df.iterrows():
        i = data_types.index(row['group1'])
        j = data_types.index(row['group2'])
        d_matrix[i, j] = row['effect_size']
        d_matrix[j, i] = -row['effect_size']  # Asymmetric (effect of group1 - group2)
    
    # Create the heatmap
    sns.heatmap(
        d_matrix,
        annot=True,
        fmt='.4f',
        cmap='PiYG',
        xticklabels=data_types,
        yticklabels=data_types,
        center=0,
        cbar_kws={'label': 'Effect Size (Cohen\'s d)'}
    )
    
    plt.title(f'XGBoost-PCA: Post-hoc Test Effect Sizes for {metric.upper()}', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgboost_pca_posthoc_effectsizes_{metric}.png'), dpi=300)
    plt.close()
    
    # 3. Create bar plot of mean differences
    plt.figure(figsize=(10, 6))
    
    # Create a combined group label
    posthoc_df['comparison'] = posthoc_df['group1'] + ' vs ' + posthoc_df['group2']
    
    # Sort by absolute mean difference
    posthoc_df = posthoc_df.assign(abs_mean_diff=posthoc_df['mean_diff'].abs()).sort_values('abs_mean_diff', ascending=False)
    
    # Create the bar chart
    ax = sns.barplot(
        x='comparison',
        y='mean_diff',
        data=posthoc_df
    )
    
    # Add significance asterisks
    for i, row in posthoc_df.reset_index().iterrows():
        sig = '***' if row['bonferroni_p'] < 0.001 else '**' if row['bonferroni_p'] < 0.01 else '*' if row['bonferroni_p'] < 0.05 else ''
        if sig:
            ax.text(i, row['mean_diff'] + (0.02 if row['mean_diff'] >= 0 else -0.04), 
                   sig, ha='center', fontsize=12)
    
    # Add a horizontal line at zero
    plt.axhline(y=0, color='gray', linestyle='--')
    
    plt.title(f'XGBoost-PCA: Mean Differences in {metric.upper()} Between Data Types', fontsize=14)
    plt.ylabel(f'Mean Difference in {metric.upper()}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgboost_pca_posthoc_meandiffs_{metric}.png'), dpi=300)
    plt.close()

def create_anova_visualizations(anova_df):
    """
    Create visualizations for ANOVA results
    """
    if anova_df.empty:
        return
    
    # Create a bar chart of F-statistics
    plt.figure(figsize=(10, 6))
    
    # Sort by F-statistic
    anova_df = anova_df.sort_values('f_statistic', ascending=False)
    
    # Create the bar chart
    ax = sns.barplot(
        x='metric',
        y='f_statistic',
        data=anova_df
    )
    
    # Add significance asterisks
    for i, row in anova_df.reset_index().iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        if sig:
            ax.text(i, row['f_statistic'] + 0.5, 
                   sig, ha='center', fontsize=12)
    
    plt.title('XGBoost-PCA: ANOVA F-Statistics by Metric', fontsize=14)
    plt.ylabel('F-Statistic', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_pca_anova_fstatistics.png'), dpi=300)
    plt.close()
    
    # Create a bar chart of effect sizes
    plt.figure(figsize=(10, 6))
    
    # Sort by effect size
    anova_df = anova_df.sort_values('effect_size', ascending=False)
    
    # Create the bar chart
    ax = sns.barplot(
        x='metric',
        y='effect_size',
        data=anova_df
    )
    
    # Add significance asterisks
    for i, row in anova_df.reset_index().iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        if sig:
            ax.text(i, row['effect_size'] + 0.02, 
                   sig, ha='center', fontsize=12)
    
    plt.title('XGBoost-PCA: ANOVA Effect Sizes by Metric', fontsize=14)
    plt.ylabel('Effect Size (η²)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_pca_anova_effectsizes.png'), dpi=300)
    plt.close()

def create_summary_visualizations(results_df, metrics):
    """
    Create summary visualizations of the results
    """
    if results_df.empty:
        return
    
    # 1. Create box plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Create the box plot
        ax = sns.boxplot(
            x='data_type',
            y=metric,
            data=results_df
        )
        
        # Add individual data points
        sns.stripplot(
            x='data_type',
            y=metric,
            data=results_df,
            color='black',
            alpha=0.5,
            jitter=True
        )
        
        # Calculate and display means
        means = results_df.groupby('data_type')[metric].mean()
        for i, data_type in enumerate(means.index):
            ax.text(i, means[data_type] + 0.02, 
                   f'{means[data_type]:.3f}', ha='center', fontsize=10)
        
        plt.title(f'XGBoost-PCA: {metric.upper()} by Data Type', fontsize=14)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(OUTPUT_DIR, f'xgboost_pca_boxplot_{metric}.png'), dpi=300)
        plt.close()
    
    # 2. Create a combined visualization of all metrics
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each metric
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        # Create the bar plot
        ax = sns.barplot(
            x='data_type',
            y=metric,
            data=results_df
        )
        
        # Add individual data points
        sns.stripplot(
            x='data_type',
            y=metric,
            data=results_df,
            color='black',
            alpha=0.5,
            jitter=True
        )
        
        # Calculate and display means
        means = results_df.groupby('data_type')[metric].mean()
        for j, data_type in enumerate(means.index):
            ax.text(j, means[data_type] + 0.02, 
                   f'{means[data_type]:.3f}', ha='center', fontsize=9)
        
        plt.title(metric.upper(), fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle('XGBoost-PCA: Performance Metrics by Data Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_pca_combined_metrics.png'), dpi=300)
    plt.close()
    
    # 3. Create a heatmap of mean metrics
    plt.figure(figsize=(10, 8))
    
    # Calculate mean metrics by data type
    means = results_df.groupby('data_type')[metrics].mean()
    
    # Create the heatmap
    sns.heatmap(
        means,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        linewidths=.5,
        cbar_kws={'label': 'Value'}
    )
    
    plt.title('XGBoost-PCA: Mean Performance Metrics by Data Type', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_pca_metrics_heatmap.png'), dpi=300)
    plt.close()

def main():
    """
    Main function to analyze XGBoost-PCA model across data types
    """
    print("Collecting XGBoost-PCA results...")
    xgb_pca_results = collect_xgboost_pca_results()
    
    # Save results to output directory
    if not xgb_pca_results.empty:
        xgb_pca_results.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_pca_results.csv'), index=False)
        print(f"Saved XGBoost-PCA results with {len(xgb_pca_results)} rows to {OUTPUT_DIR}")
        
        # Print summary
        summary = xgb_pca_results.groupby('data_type').size().reset_index(name='count')
        print("\nSummary of collected results:")
        print(summary)
        
        # Also save the summary to the output directory
        summary.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_pca_summary_count.csv'), index=False)
        
        # Perform statistical analysis
        perform_statistical_analysis(xgb_pca_results)
        
        print("\nAnalysis complete!")
    else:
        print("\nNo results data available for analysis")

if __name__ == "__main__":
    main()