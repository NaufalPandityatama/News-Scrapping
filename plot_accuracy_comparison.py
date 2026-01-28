import matplotlib.pyplot as plt
import numpy as np

# Data extracted from CNN+K-Fold+SMOTE.ipynb
classes = ['Class 0', 'Class 1', 'Class 2']

# Experiment 1 (No K-Fold) - Recall values
exp1_recall = [0.75, 0.69, 0.55]

# Experiment 2 (K-Fold, Imbalanced) - Average Recall values
exp2_recall = [0.7063, 0.6987, 0.4820]

# Experiment 3 (K-Fold & SMOTE) - Average Recall values
exp3_recall = [0.6668, 0.5738, 0.5924]

x = np.arange(len(classes))
width = 0.25

# Setting up the plot with premium colors and styling
plt.style.use('ggplot') # Using a clean style
fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width, exp1_recall, width, label='Exp 1: No K-Fold', color='#3498db', alpha=0.85)
rects2 = ax.bar(x, exp2_recall, width, label='Exp 2: K-Fold Imbalanced', color='#e74c3c', alpha=0.85)
rects3 = ax.bar(x + width, exp3_recall, width, label='Exp 3: K-Fold + SMOTE', color='#2ecc71', alpha=0.85)

# Formatting
ax.set_ylabel('Recall (Accuracy per Class)', fontsize=12, fontweight='bold')
ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Class-wise Performance Across Experiments\n(Accuracy of Class 0, 1, and 2)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()

# Save the plot
output_path = 'class_accuracy_comparison.png'
plt.savefig(output_path, dpi=300)
print(f"Graph saved to {output_path}")
plt.show()
