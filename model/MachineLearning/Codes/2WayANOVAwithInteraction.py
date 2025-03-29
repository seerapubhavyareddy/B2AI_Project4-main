import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuration - use the same as your original code
BASE_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features"
OUTPUT_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/two_way_anova_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_TYPES = ['fimo', 'deep', 'rp', 'reg']
PCA_COMPONENT = "0.95"
CFS_THRESHOLD = "0.7"
PCA_FEATURES = "120"
CFS_FEATURES = "200"
FOLDS = range(1, 6)  # Folds 1-5

# Use your existing functions for collecting data
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

def perform_manual_two_way_anova(results_df):
    """
    Perform a manual two-way ANOVA with interaction using scipy.stats functions
    This avoids the need for statsmodels
    """
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    anova_results = []
    
    print("\n==== TWO-WAY ANOVA WITH INTERACTIONS ====")
    
    for metric in metrics:
        print(f"\nAnalyzing {metric.upper()}:")
        
        # Get unique factors
        model_types = results_df['model_type'].unique()
        feature_selections = results_df['feature_selection'].unique()
        breathing_types = results_df['data_type'].unique()
        
        # Create a results dictionary to store all ANOVA results
        anova_dict = {
            'metric': metric,
            'factors': [],
            'F_values': [],
            'p_values': [],
            'significance': [],
            'effect_sizes': []
        }
        
        # 1. Main effect: Model Type
        model_groups = []
        for mt in model_types:
            model_groups.append(results_df[results_df['model_type'] == mt][metric].values)
        
        if len(model_groups) > 1:  # Only perform ANOVA if there are at least 2 groups
            F_model, p_model = stats.f_oneway(*model_groups)
            
            # Calculate effect size (Eta-squared)
            # Formula: SSbetween / SStotal
            grand_mean = results_df[metric].mean()
            ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in model_groups)
            ss_total = sum((results_df[metric] - grand_mean) ** 2)
            eta_sq_model = ss_between / ss_total if ss_total > 0 else 0
            
            anova_dict['factors'].append('Model Type')
            anova_dict['F_values'].append(F_model)
            anova_dict['p_values'].append(p_model)
            anova_dict['significance'].append(p_model < 0.05)
            anova_dict['effect_sizes'].append(eta_sq_model)
            
            print(f"Model Type: F={F_model:.4f}, p={p_model:.4f}, eta_sq={eta_sq_model:.4f}")
        
        # 2. Main effect: Feature Selection
        feature_groups = []
        for fs in feature_selections:
            feature_groups.append(results_df[results_df['feature_selection'] == fs][metric].values)
        
        if len(feature_groups) > 1:
            F_feature, p_feature = stats.f_oneway(*feature_groups)
            
            # Calculate effect size
            ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in feature_groups)
            eta_sq_feature = ss_between / ss_total if ss_total > 0 else 0
            
            anova_dict['factors'].append('Feature Selection')
            anova_dict['F_values'].append(F_feature)
            anova_dict['p_values'].append(p_feature)
            anova_dict['significance'].append(p_feature < 0.05)
            anova_dict['effect_sizes'].append(eta_sq_feature)
            
            print(f"Feature Selection: F={F_feature:.4f}, p={p_feature:.4f}, eta_sq={eta_sq_feature:.4f}")
        
        # 3. Main effect: Breathing Type
        breath_groups = []
        for bt in breathing_types:
            breath_groups.append(results_df[results_df['data_type'] == bt][metric].values)
        
        if len(breath_groups) > 1:
            F_breath, p_breath = stats.f_oneway(*breath_groups)
            
            # Calculate effect size
            ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in breath_groups)
            eta_sq_breath = ss_between / ss_total if ss_total > 0 else 0
            
            anova_dict['factors'].append('Breathing Type')
            anova_dict['F_values'].append(F_breath)
            anova_dict['p_values'].append(p_breath)
            anova_dict['significance'].append(p_breath < 0.05)
            anova_dict['effect_sizes'].append(eta_sq_breath)
            
            print(f"Breathing Type: F={F_breath:.4f}, p={p_breath:.4f}, eta_sq={eta_sq_breath:.4f}")
        
        # 4. Interaction: Model Type x Feature Selection
        # This is a simplified approximation of interaction effects
        interaction_groups = []
        
        for mt in model_types:
            for fs in feature_selections:
                interaction_groups.append(
                    results_df[(results_df['model_type'] == mt) & 
                              (results_df['feature_selection'] == fs)][metric].values
                )
        
        # We'll use a somewhat crude approximation for interaction effects
        # by comparing if the combined groups are significantly different
        if len(interaction_groups) > 1:
            F_model_feature, p_model_feature = stats.f_oneway(*interaction_groups)
            
            # A rough approximation of interaction effect
            # If this p-value is significant but different from both main effects,
            # we can infer an interaction
            interaction_effect = (p_model_feature < 0.05 and 
                                 (p_model_feature != p_model) and 
                                 (p_model_feature != p_feature))
            
            anova_dict['factors'].append('Model × Feature')
            anova_dict['F_values'].append(F_model_feature)
            anova_dict['p_values'].append(p_model_feature)
            anova_dict['significance'].append(interaction_effect)
            anova_dict['effect_sizes'].append(None)  # Complex to calculate accurately
            
            print(f"Model × Feature: F={F_model_feature:.4f}, p={p_model_feature:.4f}")
        
        # 5. Interaction: Model Type x Breathing Type
        interaction_groups = []
        
        for mt in model_types:
            for bt in breathing_types:
                interaction_groups.append(
                    results_df[(results_df['model_type'] == mt) & 
                              (results_df['data_type'] == bt)][metric].values
                )
        
        if len(interaction_groups) > 1:
            F_model_breath, p_model_breath = stats.f_oneway(*interaction_groups)
            
            # Rough approximation of interaction
            interaction_effect = (p_model_breath < 0.05 and 
                                 (p_model_breath != p_model) and 
                                 (p_model_breath != p_breath))
            
            anova_dict['factors'].append('Model × Breathing')
            anova_dict['F_values'].append(F_model_breath)
            anova_dict['p_values'].append(p_model_breath)
            anova_dict['significance'].append(interaction_effect)
            anova_dict['effect_sizes'].append(None)
            
            print(f"Model × Breathing: F={F_model_breath:.4f}, p={p_model_breath:.4f}")
        
        # 6. Interaction: Feature Selection x Breathing Type
        interaction_groups = []
        
        for fs in feature_selections:
            for bt in breathing_types:
                interaction_groups.append(
                    results_df[(results_df['feature_selection'] == fs) & 
                              (results_df['data_type'] == bt)][metric].values
                )
        
        if len(interaction_groups) > 1:
            F_feature_breath, p_feature_breath = stats.f_oneway(*interaction_groups)
            
            # Rough approximation of interaction
            interaction_effect = (p_feature_breath < 0.05 and 
                                 (p_feature_breath != p_feature) and 
                                 (p_feature_breath != p_breath))
            
            anova_dict['factors'].append('Feature × Breathing')
            anova_dict['F_values'].append(F_feature_breath)
            anova_dict['p_values'].append(p_feature_breath)
            anova_dict['significance'].append(interaction_effect)
            anova_dict['effect_sizes'].append(None)
            
            print(f"Feature × Breathing: F={F_feature_breath:.4f}, p={p_feature_breath:.4f}")
        
        # Optional: Triple interaction (Model × Feature × Breathing)
        # This gets quite complex to calculate properly without statsmodels
        
        # Save ANOVA results in a dataframe for this metric
        metric_results = pd.DataFrame({
            'factor': anova_dict['factors'],
            'F_value': anova_dict['F_values'],
            'p_value': anova_dict['p_values'],
            'significant': anova_dict['significance'],
            'effect_size': anova_dict['effect_sizes']
        })
        metric_results['metric'] = metric
        
        anova_results.append(metric_results)
        
        # Visualize the results with improved charts
        create_visualizations(results_df, metric, anova_dict)
    
    # Combine all results
    all_anova_results = pd.concat(anova_results)
    all_anova_results.to_csv(os.path.join(OUTPUT_DIR, 'two_way_anova_results.csv'), index=False)
    
    # Create combined visualizations
    create_heatmap_visualizations(all_anova_results)
    
    return all_anova_results

def create_visualizations(results_df, metric, anova_dict):
    """
    Create visualizations for a specific metric with error bars and significance indicators
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get unique factors
    model_types = results_df['model_type'].unique()
    feature_selections = results_df['feature_selection'].unique()
    breathing_types = results_df['data_type'].unique()
    
    # Find indexes of factors in the results
    try:
        idx_model = anova_dict['factors'].index('Model Type')
        p_model = anova_dict['p_values'][idx_model]
        sig_model = '*' if p_model < 0.05 else 'ns'
    except ValueError:
        p_model = 1.0
        sig_model = 'ns'
        
    try:
        idx_feature = anova_dict['factors'].index('Feature Selection')
        p_feature = anova_dict['p_values'][idx_feature]
        sig_feature = '*' if p_feature < 0.05 else 'ns'
    except ValueError:
        p_feature = 1.0
        sig_feature = 'ns'
    
    try:
        idx_breath = anova_dict['factors'].index('Breathing Type')
        p_breath = anova_dict['p_values'][idx_breath]
        sig_breath = '*' if p_breath < 0.05 else 'ns'
    except ValueError:
        p_breath = 1.0
        sig_breath = 'ns'
    
    # 1. Model Type vs Feature Selection
    ax1 = axes[0, 0]
    model_feature_data = results_df.groupby(['model_type', 'feature_selection']).agg({
        metric: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-level columns
    model_feature_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in model_feature_data.columns]
    
    # Calculate 95% confidence intervals
    model_feature_data[f'{metric}_ci'] = model_feature_data.apply(
        lambda x: 1.96 * (x[f'{metric}_std'] / np.sqrt(x[f'{metric}_count'])),
        axis=1
    )
    
    # Plot with manual error bars
    bar_width = 0.35
    colors = plt.cm.Set2.colors
    
    for i, fs in enumerate(feature_selections):
        # Get data for this feature selection
        subset = model_feature_data[model_feature_data['feature_selection'] == fs]
        
        # Calculate positions
        positions = np.arange(len(model_types)) + (i - 0.5 * (len(feature_selections) - 1)) * bar_width * 0.8
        
        # Get heights
        heights = [subset[subset['model_type'] == mt][f'{metric}_mean'].values[0] if not subset[subset['model_type'] == mt].empty else 0 for mt in model_types]
        errors = [subset[subset['model_type'] == mt][f'{metric}_ci'].values[0] if not subset[subset['model_type'] == mt].empty else 0 for mt in model_types]
        
        # Plot bars
        bars = ax1.bar(positions, heights, bar_width * 0.7, label=fs, color=colors[i])
        
        # Add error bars
        ax1.errorbar(positions, heights, yerr=errors, fmt='none', color='black', capsize=5)
    
    # Annotate with significance
    ax1.text(0.5, 1.05, f"Model: {sig_model} (p={p_model:.4f}), Feature: {sig_feature} (p={p_feature:.4f})", 
           transform=ax1.transAxes, ha='center', fontsize=10)
    
    ax1.set_title(f'{metric.capitalize()} by Model Type and Feature Selection')
    ax1.set_xticks(np.arange(len(model_types)))
    ax1.set_xticklabels(model_types)
    ax1.set_ylim(0.75, 1.0)
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel(metric.capitalize())
    ax1.legend(title="Feature Selection")
    
    # 2. Model Type vs Breathing Type
    ax2 = axes[0, 1]
    model_breath_data = results_df.groupby(['model_type', 'data_type']).agg({
        metric: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-level columns
    model_breath_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in model_breath_data.columns]
    
    # Calculate 95% confidence intervals
    model_breath_data[f'{metric}_ci'] = model_breath_data.apply(
        lambda x: 1.96 * (x[f'{metric}_std'] / np.sqrt(x[f'{metric}_count'])),
        axis=1
    )
    
    for i, bt in enumerate(breathing_types):
        # Get data for this breathing type
        subset = model_breath_data[model_breath_data['data_type'] == bt]
        
        # Calculate positions
        positions = np.arange(len(model_types)) + (i - 0.5 * (len(breathing_types) - 1)) * bar_width * 0.8
        
        # Get heights
        heights = [subset[subset['model_type'] == mt][f'{metric}_mean'].values[0] if not subset[subset['model_type'] == mt].empty else 0 for mt in model_types]
        errors = [subset[subset['model_type'] == mt][f'{metric}_ci'].values[0] if not subset[subset['model_type'] == mt].empty else 0 for mt in model_types]
        
        # Plot bars
        bars = ax2.bar(positions, heights, bar_width * 0.7, label=bt, color=colors[i+2])
        
        # Add error bars
        ax2.errorbar(positions, heights, yerr=errors, fmt='none', color='black', capsize=5)
    
    # Annotate with significance
    ax2.text(0.5, 1.05, f"Model: {sig_model} (p={p_model:.4f}), Breathing: {sig_breath} (p={p_breath:.4f})", 
           transform=ax2.transAxes, ha='center', fontsize=10)
    
    ax2.set_title(f'{metric.capitalize()} by Model Type and Breathing Type')
    ax2.set_xticks(np.arange(len(model_types)))
    ax2.set_xticklabels(model_types)
    ax2.set_ylim(0.75, 1.0)
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel(metric.capitalize())
    ax2.legend(title="Breathing Type")
    
    # 3. Feature Selection vs Breathing Type
    ax3 = axes[1, 0]
    feature_breath_data = results_df.groupby(['feature_selection', 'data_type']).agg({
        metric: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-level columns
    feature_breath_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in feature_breath_data.columns]
    
    # Calculate 95% confidence intervals
    feature_breath_data[f'{metric}_ci'] = feature_breath_data.apply(
        lambda x: 1.96 * (x[f'{metric}_std'] / np.sqrt(x[f'{metric}_count'])),
        axis=1
    )
    
    for i, bt in enumerate(breathing_types):
        # Get data for this breathing type
        subset = feature_breath_data[feature_breath_data['data_type'] == bt]
        
        # Calculate positions
        positions = np.arange(len(feature_selections)) + (i - 0.5 * (len(breathing_types) - 1)) * bar_width * 0.8
        
        # Get heights
        heights = [subset[subset['feature_selection'] == fs][f'{metric}_mean'].values[0] if not subset[subset['feature_selection'] == fs].empty else 0 for fs in feature_selections]
        errors = [subset[subset['feature_selection'] == fs][f'{metric}_ci'].values[0] if not subset[subset['feature_selection'] == fs].empty else 0 for fs in feature_selections]
        
        # Plot bars
        bars = ax3.bar(positions, heights, bar_width * 0.7, label=bt, color=colors[i+2])
        
        # Add error bars
        ax3.errorbar(positions, heights, yerr=errors, fmt='none', color='black', capsize=5)
    
    # Annotate with significance
    ax3.text(0.5, 1.05, f"Feature: {sig_feature} (p={p_feature:.4f}), Breathing: {sig_breath} (p={p_breath:.4f})", 
           transform=ax3.transAxes, ha='center', fontsize=10)
    
    ax3.set_title(f'{metric.capitalize()} by Feature Selection and Breathing Type')
    ax3.set_xticks(np.arange(len(feature_selections)))
    ax3.set_xticklabels(feature_selections)
    ax3.set_ylim(0.75, 1.0)
    ax3.set_xlabel('Feature Selection')
    ax3.set_ylabel(metric.capitalize())
    ax3.legend(title="Breathing Type")
    
    # 4. Interaction Plot for All Three Factors
    ax4 = axes[1, 1]
    interaction_data = results_df.groupby(['model_type', 'feature_selection', 'data_type']).agg({
        metric: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-level columns
    interaction_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in interaction_data.columns]
    
    # Calculate 95% confidence intervals
    interaction_data[f'{metric}_ci'] = interaction_data.apply(
        lambda x: 1.96 * (x[f'{metric}_std'] / np.sqrt(x[f'{metric}_count'])),
        axis=1
    )
    
    # Create model_feature combinations
    interaction_data['combo'] = interaction_data['model_type'] + '_' + interaction_data['feature_selection']
    
    # Plot interaction lines
    for combo in interaction_data['combo'].unique():
        subset = interaction_data[interaction_data['combo'] == combo]
        subset = subset.sort_values('data_type')  # Sort for proper line drawing
        
        ax4.errorbar(
            subset['data_type'], 
            subset[f'{metric}_mean'],
            yerr=subset[f'{metric}_ci'],
            marker='o',
            capsize=5,
            label=combo
        )
    
    ax4.set_title(f'Interaction Effect on {metric.capitalize()}')
    ax4.set_ylim(0.75, 1.0)
    ax4.set_xlabel('Breathing Type')
    ax4.set_ylabel(metric.capitalize())
    ax4.legend(title="Model & Feature", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'two_way_anova_{metric}.png'), dpi=300)
    plt.close()

def create_heatmap_visualizations(all_anova_results):
    """
    Create heatmap visualizations of p-values and effect sizes
    """
    # Create p-value heatmap
    plt.figure(figsize=(12, 8))
    
    # Prepare data - replace NaN with 1.0 (no effect) and truncate very small p-values
    p_values = all_anova_results.pivot_table(
        index='factor', 
        columns='metric', 
        values='p_value',
        aggfunc='first'
    ).fillna(1.0)
    
    # Format annotations to show p-values and significance
    def format_p_value(p):
        if p < 0.001:
            return f"{p:.4f}***"
        elif p < 0.01:
            return f"{p:.4f}**"
        elif p < 0.05:
            return f"{p:.4f}*"
        else:
            return f"{p:.4f}"
    
    # Apply formatting to p-values
    p_annotations = p_values.applymap(format_p_value)
    
    # Create heatmap
    sns.heatmap(
        p_values, 
        annot=p_annotations, 
        cmap='coolwarm_r', 
        vmin=0, 
        vmax=0.1,
        fmt='',
        linewidths=.5, 
        cbar_kws={'label': 'p-value'}
    )
    
    plt.title('Two-way ANOVA p-values')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'two_way_anova_pvalues_heatmap.png'), dpi=300)
    plt.close()
    
    # Create effect size heatmap (only for main effects that have effect size calculations)
    effect_size_data = all_anova_results[~all_anova_results['effect_size'].isna()]
    
    if not effect_size_data.empty:
        plt.figure(figsize=(12, 8))
        
        effect_sizes = effect_size_data.pivot_table(
            index='factor', 
            columns='metric', 
            values='effect_size',
            aggfunc='first'
        ).fillna(0)
        
        # Create heatmap
        sns.heatmap(
            effect_sizes, 
            annot=True, 
            cmap='viridis', 
            vmin=0, 
            vmax=1,
            fmt='.3f',
            linewidths=.5, 
            cbar_kws={'label': 'Effect Size (η²)'}
        )
        
        plt.title('Two-way ANOVA Effect Sizes (η²)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'two_way_anova_effect_sizes_heatmap.png'), dpi=300)
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
    
    # Run manual two-way ANOVA with interactions
    anova_results = perform_manual_two_way_anova(all_results)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()