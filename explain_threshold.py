import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Probability distribution
ax1 = axes[0, 0]
np.random.seed(42)
legit_probs = np.random.beta(2, 5, 1000)  # Legitimate skewed left
fraud_probs = np.random.beta(5, 2, 100)   # Fraud skewed right

ax1.hist(legit_probs, bins=50, alpha=0.7, label='Legitimate (1000)', color='blue', density=True)
ax1.hist(fraud_probs, bins=50, alpha=0.7, label='Fraud (100)', color='red', density=True)
ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
ax1.axvline(x=0.97, color='green', linestyle='--', linewidth=2, label='Threshold = 0.97 (Optimal)')
ax1.set_xlabel('Predicted Probability', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Model Predicted Probabilities', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Threshold impact on classification
ax2 = axes[0, 1]
thresholds = [0.3, 0.5, 0.7, 0.97]
true_positives = [98, 92, 90, 87]
false_positives = [1000, 100, 50, 20]

x = np.arange(len(thresholds))
width = 0.35

bars1 = ax2.bar(x - width/2, true_positives, width, label='Fraud Caught (%)', color='green', alpha=0.7)
bars2 = ax2.bar(x + width/2, false_positives, width, label='False Alarms', color='red', alpha=0.7)

ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Impact of Different Thresholds', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{t}' for t in thresholds])
ax2.set_xlabel('Threshold Value', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

# 3. Decision boundary visualization
ax3 = axes[1, 0]
probabilities = np.linspace(0, 1, 100)

# Simulate predictions at different thresholds
threshold_50 = ['Legit' if p < 0.5 else 'Fraud' for p in probabilities]
threshold_97 = ['Legit' if p < 0.97 else 'Fraud' for p in probabilities]

colors_50 = ['blue' if c == 'Legit' else 'red' for c in threshold_50]
colors_97 = ['blue' if c == 'Legit' else 'red' for c in threshold_97]

ax3.scatter(probabilities, [0.3]*100, c=colors_50, alpha=0.6, s=50, label='Threshold 0.5')
ax3.scatter(probabilities, [0.7]*100, c=colors_97, alpha=0.6, s=50, label='Threshold 0.97')
ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax3.axvline(x=0.97, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_xlabel('Predicted Probability', fontsize=12)
ax3.set_ylabel('Threshold Type', fontsize=12)
ax3.set_yticks([0.3, 0.7])
ax3.set_yticklabels(['Default (0.5)', 'Optimal (0.97)'])
ax3.set_title('Decision Boundaries', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add legend with color explanation
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='Classified as Legitimate'),
                   Patch(facecolor='red', label='Classified as Fraud')]
ax3.legend(handles=legend_elements, fontsize=10)

# 4. Metrics comparison
ax4 = axes[1, 1]
metrics_data = {
    'Threshold': ['0.5\n(Default)', '0.97\n(Optimal)'],
    'Precision': [0.375, 0.680],
    'Recall': [0.923, 0.872],
    'F1 Score': [0.533, 0.764]
}

x = np.arange(len(metrics_data['Threshold']))
width = 0.25

bars1 = ax4.bar(x - width, metrics_data['Precision'], width, label='Precision', color='skyblue')
bars2 = ax4.bar(x, metrics_data['Recall'], width, label='Recall', color='lightcoral')
bars3 = ax4.bar(x + width, metrics_data['F1 Score'], width, label='F1 Score', color='lightgreen')

ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Metric Comparison: Default vs Optimal Threshold', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_data['Threshold'])
ax4.legend(fontsize=10)
ax4.set_ylim(0, 1.0)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.suptitle('Understanding Classification Thresholds', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('threshold_explanation.png', dpi=300, bbox_inches='tight')
print("✅ Threshold explanation diagram saved to 'threshold_explanation.png'")
plt.show()

# Print explanation
print("\n" + "="*70)
print("  THRESHOLD EXPLANATION")
print("="*70)
print("\n📌 What is a threshold?")
print("   A cutoff point to convert probabilities into binary predictions.")
print("\n📌 Default threshold (0.5):")
print("   - Treats both classes equally")
print("   - Catches 92% of fraud")
print("   - Has 100 false alarms")
print("   - F1 Score: 0.533")
print("\n📌 Optimal threshold (0.97):")
print("   - Optimized for F1 score")
print("   - Catches 87% of fraud (slightly lower)")
print("   - Only 20 false alarms (80% reduction!)")
print("   - F1 Score: 0.764 (43% improvement!)")
print("\n📌 Why is 0.97 better?")
print("   - Better balance between catching fraud and avoiding false alarms")
print("   - Maximizes the F1 score metric")
print("   - More practical for real-world deployment")
print("="*70)
