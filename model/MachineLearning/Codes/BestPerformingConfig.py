import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the output directory
OUTPUT_DIR = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/additional_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the detailed performance data
csv_path = "/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features/anova_results_dataTypes/detailed_performance_by_breathing_type.csv"
data = pd.read_csv(csv_path)

# First, let's print the column names to see what we're working with
print("Columns in the dataset:", data.columns.tolist())

# Let's look at the first few rows to understand the structure
print("\nFirst few rows of the dataset:")
print(data.head())

# Assuming 'model' column contains both model type and feature selection (e.g., 'RF_CFS')
# Extract model type and feature selection from 'model' column if needed
if 'model' in data.columns and 'model_type' not in data.columns:
    data['model_type'] = data['model'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    data['feature_selection'] = data['model'].apply(lambda x: x.split('_')[1] if '_' in x else '')
    print("\nExtracted model_type and feature_selection from model column")

# Now identify the best configuration based on AUC (changed from accuracy)
# If specific columns aren't available, work with what we have
metrics_to_check = ['auc_mean', 'auc_std', 'accuracy_mean', 'f1_mean', 'precision_mean', 'recall_mean']
available_metrics = [col for col in metrics_to_check if col in data.columns]

if not available_metrics:
    # If we don't have _mean columns, try without the _mean suffix
    metrics_to_check = ['auc', 'accuracy', 'f1', 'precision', 'recall']
    available_metrics = [col for col in metrics_to_check if col in data.columns]

print(f"\nUsing these metrics: {available_metrics}")

# Find the AUC metrics (both mean and std)
auc_mean_metric = next((m for m in available_metrics if 'auc' in m.lower() and 'mean' in m.lower()), None)
if not auc_mean_metric:
    auc_mean_metric = next((m for m in available_metrics if 'auc' in m.lower() and 'std' not in m.lower()), None)

auc_std_metric = next((m for m in available_metrics if 'auc' in m.lower() and 'std' in m.lower()), None)
if not auc_std_metric and 'auc_mean' in data.columns:
    # Try to find column with standard deviation
    possible_std_cols = [col for col in data.columns if 'auc' in col.lower() and 'std' in col.lower()]
    if possible_std_cols:
        auc_std_metric = possible_std_cols[0]

print(f"AUC mean metric: {auc_mean_metric}")
print(f"AUC std metric: {auc_std_metric}")

# Try to find the best configuration
try:
    # First attempt: Using model_type and feature_selection if available
    if 'model_type' in data.columns and 'feature_selection' in data.columns:
        best_row_idx = data[data['data_type'] == 'reg'][auc_mean_metric].idxmax()
        best_config = data.iloc[[best_row_idx]]
        print("\nBest configuration found using model_type and feature_selection")
    else:
        # Second attempt: Just use data_type and model
        best_row_idx = data[data['data_type'] == 'reg'][auc_mean_metric].idxmax()
        best_config = data.iloc[[best_row_idx]]
        print("\nBest configuration found using data_type and model")
        
    print(f"Best configuration: {best_config[['data_type', 'model']].values[0]}")
    
    # Get values for all data types with the same model
    best_model = best_config['model'].values[0]
    all_data_types = []
    
    for data_type in ['reg', 'deep', 'fimo', 'rp']:
        if data_type in data['data_type'].values:
            config = data[(data['data_type'] == data_type) & (data['model'] == best_model)]
            if not config.empty:
                if auc_mean_metric and auc_std_metric:
                    values = [
                        config[auc_mean_metric].values[0],
                        config[auc_std_metric].values[0]
                    ]
                else:
                    values = [config[auc_mean_metric].values[0], 0]  # Default std=0 if not available
                all_data_types.append({
                    'data_type': data_type,
                    'values': values
                })
    
    print(f"\nFound data for {len(all_data_types)} data types with model {best_model}")
    

    # Create a heatmap showing AUC performance across data types and models with standard deviation
    def create_auc_heatmap_with_std():
        """
        Create a heatmap showing AUC performance with standard deviation
        """
        try:
            if 'model_type' in data.columns and 'feature_selection' in data.columns:
                pivot_mean = data.pivot_table(
                    index=['data_type'], 
                    columns=['model_type', 'feature_selection'],
                    values='auc_mean'
                )
                
                # Create a pivot table for std values
                # pivot_std = data.pivot_table(
                #     index=['data_type'], 
                #     columns=['model_type', 'feature_selection'],
                #     values='auc_std'
                # )
            else:
                pivot_mean = data.pivot_table(
                    index=['data_type'], 
                    columns=['model'],
                    values='auc_mean'
                )
                
                # # Create a pivot table for std values
                # pivot_std = data.pivot_table(
                #     index=['data_type'], 
                #     columns=['model'],
                #     values='auc_std'
                # )
            
            # Set up the figure
            plt.figure(figsize=(12, 8))
            
            # Create the heatmap with mean values
            ax = sns.heatmap(
                pivot_mean, 
                annot=False,  # We'll add custom annotations with std
                cmap='viridis', 
                linewidths=.5,
                cbar_kws={'label': 'AUC'}
            )
            
            # Add custom annotations with std deviation
            for i in range(len(pivot_mean.index)):
                for j in range(len(pivot_mean.columns)):
                    mean_val = pivot_mean.iloc[i, j]
                    # std_val = pivot_std.iloc[i, j]
                    # Format the annotation to show mean±std
                    # text = f"{mean_val:.3f}±{std_val:.3f}" if not np.isnan(mean_val) else "N/A"
                    text = f"{mean_val:.3f}" if not np.isnan(mean_val) else "N/A"
                    ax.text(j + 0.5, i + 0.5, text,
                        ha="center", va="center", color="white" if mean_val < 0.85 else "black")
            
            # Customize the chart
            plt.title('AUC Across All Configurations (with Standard Deviation)', size=15)
            plt.tight_layout()
            
            # Save the figure
            heatmap_file = os.path.join(OUTPUT_DIR, 'auc_heatmap.png')
            plt.savefig(heatmap_file, dpi=300)
            plt.close()
            
            print(f"Created AUC heatmap with standard deviation: {heatmap_file}")
            return heatmap_file
        except Exception as e:
            print(f"Error creating AUC heatmap: {e}")
            return None


except Exception as e:
    print(f"Error during processing: {e}")
    
    # Attempt to create a simple visualization of the data we have
    print("\nFalling back to simple visualization...")
    
    if 'data_type' in data.columns and auc_mean_metric in data.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='data_type', y=auc_mean_metric, data=data)
        plt.title(f'AUC by Data Type')
        plt.ylim(0.6, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        simple_file = os.path.join(OUTPUT_DIR, 'simple_auc_data_type_comparison.png')
        plt.savefig(simple_file, dpi=300)
        plt.close()
        
        print(f"Created simple AUC visualization: {simple_file}")
    else:
        print("Unable to create any AUC visualizations with the available data")

# Run the AUC heatmap visualization
auc_heatmap_file = create_auc_heatmap_with_std()
print(f"\nAUC heatmap visualization created: {auc_heatmap_file}")