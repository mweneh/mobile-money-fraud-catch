import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from lightgbm import LGBMClassifier

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def engineer_features(df, is_train=True):
    """Apply advanced feature engineering"""
    df['Amount_Value_Ratio'] = df['Amount'] / (df['Value'] + 1e-6)
    df['Amount_Value_Interaction'] = df['Amount'] * df['Value']
    df['Amount_Value_Difference'] = df['Amount'] - df['Value']
    df['LogAmount'] = np.log1p(np.abs(df['Amount']))
    df['LogValue'] = np.log1p(np.abs(df['Value']))
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    df['IsLateNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    return df


print("="*70)
print("  MODEL PERFORMANCE VISUALIZATION")
print("="*70 + "\n")

# Load and preprocess data
print("📂 Loading and preprocessing data...")
train_df = pd.read_csv('training.csv')
train_df['TransactionStartTime'] = pd.to_datetime(train_df['TransactionStartTime'])

# Feature engineering
for df in [train_df]:
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['Day'] = df['TransactionStartTime'].dt.day
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Weekday'] = df['TransactionStartTime'].dt.weekday

train_df = engineer_features(train_df, is_train=True)

# Prepare features
id_cols = ['TransactionId', 'BatchId', 'SubscriptionId', 'CustomerId', 
           'CurrencyCode', 'CountryCode', 'TransactionStartTime', 'AccountId']

feature_cols = [col for col in train_df.columns 
               if col not in ['FraudResult'] + id_cols 
               and train_df[col].dtype in [np.number, 'int64', 'float64']]

X = train_df[feature_cols].fillna(0)
y = train_df['FraudResult']

# Split data (same as in training)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {X_train.shape}")
print(f"✅ Validation set: {X_val.shape}\n")

# Train the best model (LightGBM)
print("🤖 Training LightGBM model...\n")
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

# Calculate metrics
precision, recall, pr_thresholds = precision_recall_curve(y_val, y_proba)
fpr, tpr, roc_thresholds = roc_curve(y_val, y_proba)
pr_auc = auc(recall, precision)
roc_auc = auc(fpr, tpr)

print("="*70)
print("  PERFORMANCE METRICS")
print("="*70)
print(classification_report(y_val, y_pred, digits=4))
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}\n")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# 2. ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, linewidth=2, label=f'LightGBM (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Precision-Recall Curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(recall, precision, linewidth=2, label=f'LightGBM (AUC = {pr_auc:.4f})')
ax3.axhline(y=y_val.mean(), color='k', linestyle='--', linewidth=1, label='Baseline')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Feature Importance
ax4 = fig.add_subplot(gs[1, :])
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
ax4.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
ax4.set_xlabel('Importance Score')
ax4.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Prediction Distribution
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(y_proba[y_val == 0], bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
ax5.hist(y_proba[y_val == 1], bins=50, alpha=0.7, label='Fraud', color='red', density=True)
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Density')
ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Threshold Analysis
ax6 = fig.add_subplot(gs[2, 1])
# Calculate metrics at different thresholds
thresholds = np.linspace(0, 1, 100)
f1_scores = []
precisions = []
recalls = []

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    tp = ((y_pred_thresh == 1) & (y_val == 1)).sum()
    fp = ((y_pred_thresh == 1) & (y_val == 0)).sum()
    fn = ((y_pred_thresh == 0) & (y_val == 1)).sum()
    
    precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
    
    precisions.append(precision_t)
    recalls.append(recall_t)
    f1_scores.append(f1_t)

ax6.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
ax6.plot(thresholds, precisions, label='Precision', linewidth=2)
ax6.plot(thresholds, recalls, label='Recall', linewidth=2)
ax6.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)
optimal_threshold = thresholds[np.argmax(f1_scores)]
ax6.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, 
            label=f'Optimal ({optimal_threshold:.2f})')
ax6.set_xlabel('Threshold')
ax6.set_ylabel('Score')
ax6.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Class Distribution
ax7 = fig.add_subplot(gs[2, 2])
class_data = pd.DataFrame({
    'Dataset': ['Training', 'Validation'],
    'Legitimate': [len(y_train[y_train==0]), len(y_val[y_val==0])],
    'Fraud': [len(y_train[y_train==1]), len(y_val[y_val==1])]
})
x = np.arange(len(class_data))
width = 0.35
ax7.bar(x - width/2, class_data['Legitimate'], width, label='Legitimate', color='blue', alpha=0.7)
ax7.bar(x + width/2, class_data['Fraud'], width, label='Fraud', color='red', alpha=0.7)
ax7.set_ylabel('Count')
ax7.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(class_data['Dataset'])
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle('LightGBM Model Performance Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.savefig('model_performance_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to 'model_performance_visualization.png'\n")
plt.show()

# Print optimal threshold info
print("="*70)
print("  OPTIMAL THRESHOLD ANALYSIS")
print("="*70)
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"F1 Score at optimal: {max(f1_scores):.4f}")
print(f"Precision at optimal: {precisions[np.argmax(f1_scores)]:.4f}")
print(f"Recall at optimal: {recalls[np.argmax(f1_scores)]:.4f}")
print("="*70)
