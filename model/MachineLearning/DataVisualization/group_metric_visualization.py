import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Create data for each metric
metrics = ['Accuracy', 'F1-Score', 'Specificity', 'Sensitivity', 'AUC']
models = ['PCA - RF', 'CFS - RF', 'PCA - XGBOOST', 'CFS - XGBOOST']
datasets = ['FIMO', 'REG', 'RP', 'DEEP']

# Data for each metric
accuracy = {
    'PCA - RF': [0.823, 0.91, 0.722, 0.878],
    'CFS - RF': [0.864, 0.92, 0.736, 0.8614],
    'PCA - XGBOOST': [0.822, 0.932, 0.774, 0.838],
    'CFS - XGBOOST': [0.812, 0.913, 0.658, 0.866]
}

f1_score = {
    'PCA - RF': [0.841, 0.921, 0.786, 0.883],
    'CFS - RF': [0.783, 0.913, 0.69, 0.865],
    'PCA - XGBOOST': [0.854, 0.926, 0.791, 0.888],
    'CFS - XGBOOST': [0.783, 0.915, 0.678, 0.872]
}

specificity = {
    'PCA - RF': [0.839, 0.925, 0.784, 0.882],
    'CFS - RF': [0.783, 0.914, 0.683, 0.866],
    'PCA - XGBOOST': [0.856, 0.932, 0.791, 0.886],
    'CFS - XGBOOST': [0.787, 0.915, 0.673, 0.871]
}

sensitivity = {
    'PCA - RF': [0.845, 0.92, 0.793, 0.888],
    'CFS - RF': [0.798, 0.914, 0.713, 0.871],
    'PCA - XGBOOST': [0.853, 0.923, 0.794, 0.891],
    'CFS - XGBOOST': [0.792, 0.918, 0.697, 0.875]
}

auc = {
    'PCA - RF': [0.876, 0.922, 0.814, 0.893],
    'CFS - RF': [0.823, 0.908, 0.68, 0.867],
    'PCA - XGBOOST': [0.888, 0.934, 0.807, 0.898],
    'CFS - XGBOOST': [0.776, 0.889, 0.616, 0.846]
}

# Convert to DataFrames
all_metrics = [accuracy, f1_score, specificity, sensitivity, auc]
all_dfs = []

for i, metric_data in enumerate(all_metrics):
    df = pd.DataFrame(metric_data, index=datasets)
    df = df.reset_index()
    df = pd.melt(df, id_vars='index', var_name='Model', value_name=metrics[i])
    df = df.rename(columns={'index': 'Dataset'})
    all_dfs.append(df)

# Function to create grouped bar charts
def create_grouped_bar_charts(metric_dfs, metric_names):
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 5*len(metric_names)))
    
    for i, (df, metric) in enumerate(zip(metric_dfs, metric_names)):
        ax = axes[i] if len(metric_names) > 1 else axes
        sns.barplot(x='Dataset', y=metric, hue='Model', data=df, ax=ax)
        ax.set_title(f'{metric} by Model and Dataset', fontsize=16)
        ax.set_xlabel('Dataset', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.legend(title='Model', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig('grouped_bar_charts.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create heatmaps
def create_heatmaps(all_metrics, metric_names, models, datasets):
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4*len(metric_names)))
    
    for i, (metric_data, metric) in enumerate(zip(all_metrics, metric_names)):
        ax = axes[i] if len(metric_names) > 1 else axes
        
        # Create DataFrame for heatmap
        df = pd.DataFrame(metric_data, index=models)
        
        # Create heatmap
        sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f', linewidths=.5, ax=ax)
        ax.set_title(f'{metric} Heatmap', fontsize=16)
        ax.set_xlabel('Dataset', fontsize=14)
        ax.set_ylabel('Model', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create all visualizations
create_grouped_bar_charts(all_dfs, metrics)
create_heatmaps(all_metrics, metrics, models, datasets)

# Now create individual plots for each metric
# Grouped Bar Charts for each metric
for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Dataset', y=metric, hue='Model', data=df)
    plt.title(f'{metric} by Model and Dataset', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.legend(title='Model', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for p in ax.patches:
        plt.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{metric}_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

# Heatmaps for each metric
for i, (metric_data, metric) in enumerate(zip(all_metrics, metrics)):
    plt.figure(figsize=(8, 5))
    
    # Create DataFrame for heatmap
    df = pd.DataFrame(metric_data, index=models)
    
    # Create heatmap
    ax = sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
    plt.title(f'{metric} Heatmap', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{metric}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# Alternative version with a different color map and layout
for i, (metric_data, metric) in enumerate(zip(all_metrics, metrics)):
    plt.figure(figsize=(8, 5))
    
    # Create DataFrame for heatmap
    df = pd.DataFrame(metric_data, index=models)
    
    # Find min and max for consistent color scaling
    vmin = df.values.min()
    vmax = df.values.max()
    
    # Create heatmap with a different colormap
    ax = sns.heatmap(df, annot=True, cmap='RdYlGn', fmt='.3f', linewidths=.5, vmin=vmin, vmax=vmax)
    plt.title(f'{metric} Heatmap', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{metric}_heatmap_alt.png', dpi=300, bbox_inches='tight')
    plt.close()