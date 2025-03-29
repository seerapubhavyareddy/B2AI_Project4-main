import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import matplotlib.colors as mcolors
import numpy as np

# Load your CSV file
df = pd.read_csv('/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/subject_paired_ttest_results/xgboost_pca_results.csv')

# Configuration
OUTPUT_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/subject_paired_ttest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data types to compare
DATA_TYPES = ['fimo', 'deep', 'rp', 'reg']
METRICS = ['accuracy', 'f1', 'precision', 'recall', 'auc']

def perform_pairwise_ttests(df):
    """Perform pairwise t-tests between all data types for each metric."""
    results = []
    
    for metric in METRICS:
        print(f"\n===== PAIRWISE T-TESTS FOR {metric.upper()} =====")
        
        # Generate all pairs for comparison
        pairs = [(a, b) for a in DATA_TYPES for b in DATA_TYPES]
        
        for type1, type2 in pairs:
            if type1 == type2:
                # For diagonal, just add placeholder with 1.0 p-value
                results.append({
                    'metric': metric,
                    'comparison': f'{type1} vs {type2}',
                    'type1': type1,
                    'type2': type2,
                    't_statistic': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'cohens_d': 0.0,
                    'mean_diff': 0.0
                })
                continue
                
            # Get data for the two types
            data1 = df[df['data_type'] == type1][metric].values
            data2 = df[df['data_type'] == type2][metric].values
            
            # Perform t-test (independent samples)
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Welch's t-test
            
            # Calculate Cohen's d effect size
            n1, n2 = len(data1), len(data2)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            # Cohen's d
            d = abs(mean1 - mean2) / pooled_std
            
            # Mean difference
            diff = mean1 - mean2
            
            # Store results
            results.append({
                'metric': metric,
                'comparison': f'{type1} vs {type2}',
                'type1': type1,
                'type2': type2,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': d,
                'mean_diff': diff
            })
            
            # Print result (only for non-diagonal)
            if type1 != type2:
                sig_symbol = "*" if p_value < 0.05 else "ns"
                print(f"{type1} vs {type2}: t = {t_stat:.4f}, p = {p_value:.4f} {sig_symbol}, d = {d:.4f}, diff = {diff:.4f}")
    
    return pd.DataFrame(results)

def apply_bonferroni_correction(results_df):
    """Apply Bonferroni correction to p-values."""
    # Number of comparisons per metric (excluding self-comparisons)
    n_comparisons = len(DATA_TYPES) * (len(DATA_TYPES) - 1) // 2  # Only unique comparisons
    
    # Add corrected p-values
    results_df['corrected_p'] = results_df['p_value']
    
    # Only apply correction to non-diagonal entries
    mask = results_df['type1'] != results_df['type2']
    results_df.loc[mask, 'corrected_p'] = results_df.loc[mask, 'p_value'] * n_comparisons
    
    # Cap at 1.0
    results_df['corrected_p'] = results_df['corrected_p'].clip(upper=1.0)
    # Update significance based on corrected p-values
    results_df['significant_corrected'] = results_df['corrected_p'] < 0.05
    
    # Add significance symbols
    def sig_symbol(p):
        if p >= 0.05:
            return "ns"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        else:
            return "*"
    
    results_df['significance'] = results_df['corrected_p'].apply(sig_symbol)
    
    return results_df

def create_triangular_heatmap(data, metric, output_dir, value_type="pvalue"):
    """Create a triangular heatmap similar to the example image."""
    # Create a figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set titles
    if value_type == "pvalue":
        title = f'P-values for {metric.upper()} - XGBoost with PCA'
    else:
        title = f'Mean differences for {metric.upper()} - XGBoost with PCA'
    
    fig.suptitle(title, fontsize=14)
    
    # Create masks for upper and lower triangles
    mask_upper = np.triu(np.ones_like(data, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(data, dtype=bool), k=0)
    
    # For p-values (left plot): Red for significant (low p-values), blue for non-significant
    if value_type == "pvalue":
        # Create a custom colormap for p-values: Blue (high) to Red (low)
        # colors for p-values: dark blue for high (non-significant), dark red for low (significant)
        cmap_pval = mcolors.LinearSegmentedColormap.from_list(
            'custom_pval', ['#c63636', '#cb6a6a', '#d49d9d', '#dfd3d3', '#dfd3d3', '#c6cef0', '#8c9de3', '#3c5fd1'])
        
        # Left heatmap - Upper triangle with p-values
        sns.heatmap(data, ax=ax1, mask=mask_lower, cmap=cmap_pval, 
                    vmin=0, vmax=0.1, annot=True, fmt='.4f', 
                    linewidths=1, cbar_kws={'label': 'p-value'})
        ax1.set_title('p-values')
    else:
        # For mean differences (left plot): Red for negative, blue for positive
        # Create a custom colormap for mean differences: Red (negative) to Blue (positive)
        cmap_diff = mcolors.LinearSegmentedColormap.from_list(
            'custom_diff', ['#c63636', '#cb6a6a', '#d49d9d', '#dfd3d3', '#dfd3d3', '#c6cef0', '#8c9de3', '#3c5fd1'])
        
        # Determine min/max for even color scaling
        abs_max = abs(data.values).max()
        
        # Left heatmap - Upper triangle with mean differences
        sns.heatmap(data, ax=ax1, mask=mask_lower, cmap=cmap_diff, 
                    vmin=-abs_max, vmax=abs_max, annot=True, fmt='.4f', 
                    linewidths=1, cbar_kws={'label': 'Mean Difference'})
        ax1.set_title('Mean Differences')
    
    # Right heatmap - Lower triangle with p-values
    # Create corrected p-value matrix where p < 0.05 is marked with 'sig', otherwise with 'ns'
    if value_type == "pvalue":
        # Create a copy for the right plot
        sig_matrix = data.copy()
        # Apply the significance cutoff
        sig_matrix = sig_matrix.applymap(lambda x: x < 0.05)
        
        # Create a custom colormap: white for ns, red for sig
        cmap_sig = mcolors.ListedColormap(['#dfd3d3', '#c63636'])
        
        # Use a function to format the annotations
        def sig_format(val):
            return "sig" if val else "ns"
        
        sns.heatmap(sig_matrix, ax=ax2, mask=mask_upper, cmap=cmap_sig,
                    annot=sig_matrix.applymap(sig_format), fmt='', 
                    linewidths=1, cbar=False)
        ax2.set_title('Significance')
    else:
        # For Cohen's d effect sizes (right plot)
        d_matrix = data.copy()
        for i, row in enumerate(DATA_TYPES):
            for j, col in enumerate(DATA_TYPES):
                if i >= j:  # Lower triangle including diagonal
                    # Find the row in results that matches this comparison
                    matching_rows = results_df[(results_df['metric'] == metric) & 
                                              ((results_df['type1'] == row) & (results_df['type2'] == col) |
                                               (results_df['type1'] == col) & (results_df['type2'] == row))]
                    if not matching_rows.empty:
                        d_matrix.iloc[i, j] = matching_rows.iloc[0]['cohens_d']
        
        # Create a custom colormap for Cohen's d: light to dark blue
        cmap_d = mcolors.LinearSegmentedColormap.from_list(
            'custom_d', ['#dfd3d3', '#c6cef0', '#8c9de3', '#3c5fd1'])
        
        sns.heatmap(d_matrix, ax=ax2, mask=mask_upper, cmap=cmap_d,
                    vmin=0, vmax=5, annot=True, fmt='.4f', 
                    linewidths=1, cbar_kws={'label': "Cohen's d"})
        ax2.set_title("Effect Size (Cohen's d)")
    
    # Adjust labels and ticks
    for ax in [ax1, ax2]:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(DATA_TYPES, rotation=45)
        ax.set_yticklabels(DATA_TYPES, rotation=0)
    
    plt.tight_layout()
    
    # Save the figure
    if value_type == "pvalue":
        output_file = f'xgboost_pca_{metric}_pvalues_triangular.png'
    else:
        output_file = f'xgboost_pca_{metric}_diffs_triangular.png'
    
    plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_all_triangular_heatmaps(results_df):
    """Create triangular heatmaps for all metrics."""
    
    for metric in METRICS:
        # Create p-value matrix
        pvalue_matrix = pd.DataFrame(index=DATA_TYPES, columns=DATA_TYPES)
        
        # Fill the matrix with p-values
        for _, row in results_df[results_df['metric'] == metric].iterrows():
            pvalue_matrix.at[row['type1'], row['type2']] = row['corrected_p']
        
        # Convert to float type explicitly
        pvalue_matrix = pvalue_matrix.astype(float)
        
        # Create the triangular heatmap for p-values
        create_triangular_heatmap(pvalue_matrix, metric, OUTPUT_DIR, "pvalue")
        
        # Create mean difference matrix
        diff_matrix = pd.DataFrame(index=DATA_TYPES, columns=DATA_TYPES)
        
        # Fill the matrix with mean differences
        for _, row in results_df[results_df['metric'] == metric].iterrows():
            diff_matrix.at[row['type1'], row['type2']] = row['mean_diff']
        
        # Convert to float type explicitly
        diff_matrix = diff_matrix.astype(float)
        
        # Create the triangular heatmap for mean differences
        create_triangular_heatmap(diff_matrix, metric, OUTPUT_DIR, "diff")

# Main execution
try:
    results = perform_pairwise_ttests(df)
    results_df = apply_bonferroni_correction(results)

    # Save results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_pca_ttest_results.csv'), index=False)
    print(f"\nSaved detailed t-test results to {os.path.join(OUTPUT_DIR, 'xgboost_pca_ttest_results.csv')}")

    # Create triangular heatmaps
    print("\nCreating triangular heatmaps...")
    visualize_all_triangular_heatmaps(results_df)
    
    print(f"Saved triangular heatmaps to {OUTPUT_DIR}")
    print("\nAnalysis complete!")
    
except Exception as e:
    print(f"Error encountered: {e}")
