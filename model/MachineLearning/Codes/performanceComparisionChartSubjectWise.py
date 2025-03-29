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

# Print the column names to understand the data structure
print("Columns in the dataset:", data.columns.tolist())
print("\nFirst few rows of the dataset:")
print(data.head())

# Check for subject-level metrics in the data
has_subject_metrics = any('num_subjects' in col or 'num_positive' in col or 'num_negative' in col for col in data.columns)
print(f"\nSubject-level metrics found: {has_subject_metrics}")

# Determine the metric columns
if any('_mean' in col for col in data.columns):
    # Use mean columns if available
    accuracy_col = 'accuracy_mean' if 'accuracy_mean' in data.columns else 'accuracy'
    f1_col = 'f1_mean' if 'f1_mean' in data.columns else 'f1'
    precision_col = 'precision_mean' if 'precision_mean' in data.columns else 'precision'
    recall_col = 'recall_mean' if 'recall_mean' in data.columns else 'recall'
else:
    # Use direct metric columns
    accuracy_col = 'accuracy'
    f1_col = 'f1'
    precision_col = 'precision'
    recall_col = 'recall'

print(f"Using these metrics: {accuracy_col}, {f1_col}, {precision_col}, {recall_col}")

# Create a grouped bar chart like the example
def create_performance_comparison_chart():
    """
    Create a grouped bar chart comparing performance metrics across data types
    """
    try:
        # Check if we have the necessary columns
        required_cols = ['data_type', accuracy_col, f1_col, precision_col, recall_col]
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            print(f"Missing required columns: {missing}")
            return None
        
        # Group by data_type and calculate mean for each metric
        # If data already has means, this won't change the values
        grouped_data = data.groupby('data_type').agg({
            accuracy_col: 'mean',
            f1_col: 'mean',
            precision_col: 'mean',
            recall_col: 'mean'
        }).reset_index()
        
        print("Grouped data for chart:")
        print(grouped_data)
        
        # Set up the data for the grouped bar chart
        data_types = grouped_data['data_type'].tolist()
        accuracy_values = grouped_data[accuracy_col].tolist()
        f1_values = grouped_data[f1_col].tolist()
        precision_values = grouped_data[precision_col].tolist()
        recall_values = grouped_data[recall_col].tolist()
        
        # Set width of bars
        barWidth = 0.2
        
        # Set positions of the bars on X axis
        r1 = np.arange(len(data_types))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]
        
        # Create the figure with a larger size
        plt.figure(figsize=(12, 8))
        
        # Create bars
        plt.bar(r1, accuracy_values, width=barWidth, label='Accuracy', color='#483D8B')       # Dark blue/indigo
        plt.bar(r2, f1_values, width=barWidth, label='F1 Score', color='#2E8B57')             # Sea green
        plt.bar(r3, precision_values, width=barWidth, label='Precision', color='#20B2AA')      # Light sea green
        plt.bar(r4, recall_values, width=barWidth, label='Recall', color='#90EE90')           # Light green
        # Add title and axis labels
        title_suffix = "(Subject-Level Metrics)" if has_subject_metrics else ""
        plt.title(f'Performance Comparison of Best Configurations {title_suffix}', fontsize=14)
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        
        # Add xticks on the middle of the group bars
        plt.xticks([r + barWidth*1.5 for r in range(len(data_types))], data_types, rotation=45, ha='right')
        
        # Add a grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limits to match the example
        plt.ylim(0.7, 1.0)
        
        # Add a legend
        plt.legend(title='Metric', loc='upper right')
        
        # Adjust layout and save the figure
        plt.tight_layout()
        chart_file = os.path.join(OUTPUT_DIR, 'performance_comparison_chart.png')
        plt.savefig(chart_file, dpi=300)
        plt.close()
        
        print(f"Created performance comparison chart: {chart_file}")
        return chart_file
    except Exception as e:
        print(f"Error creating performance comparison chart: {e}")
        return None

# Run the visualization
chart_file = create_performance_comparison_chart()
print(f"\nPerformance comparison chart created: {chart_file}")