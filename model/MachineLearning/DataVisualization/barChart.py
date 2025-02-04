import matplotlib.pyplot as plt
import numpy as np

# Data for the table
categories = ['FIMO', 'REG', 'RP', 'DEEP']
pca_rf = [0.846, 0.92, 0.792, 0.888]
cfs_rf = [0.8714, 0.9142, 0.7136, 0.8714]
pca_xgboost = [0.852, 0.924, 0.794, 0.888]
cfs_xgboost = [0.792, 0.918, 0.698, 0.876]

# Setting the bar width and positions for each set of bars
bar_width = 0.2
index = np.arange(len(categories))

# Creating the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each row as a set of bars
bar1 = ax.bar(index - 1.5 * bar_width, pca_rf, bar_width, label='PCA - RF')
bar2 = ax.bar(index - 0.5 * bar_width, cfs_rf, bar_width, label='CFS - RF')
bar3 = ax.bar(index + 0.5 * bar_width, pca_xgboost, bar_width, label='PCA - XGBOOST')
bar4 = ax.bar(index + 1.5 * bar_width, cfs_xgboost, bar_width, label='CFS - XGBOOST')

# Adding labels, title, and customizing the chart
ax.set_xlabel('Categories')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison for Different Models and Features')
ax.set_xticks(index)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
