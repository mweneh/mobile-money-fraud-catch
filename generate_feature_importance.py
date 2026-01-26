"""
Generate Feature Importance Graph from LightGBM Model
Saves the figure to figs/feature_importance.png for use in LaTeX paper

This script replicates the exact feature engineering from the notebook
to produce matching feature importance results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# Load training data
print("Loading data...")
train = pd.read_csv('training_data.csv')

# ============================================================================
# FEATURE ENGINEERING (matching notebook exactly)
# ============================================================================
print("Engineering features...")

def engineer_features(df):
    """Apply the same feature engineering as in the notebook"""
    
    # 1. Amount and Value interactions
    df['Amount_Value_Ratio'] = df['Amount'] / (df['Value'] + 1e-6)
    df['Amount_Value_Interaction'] = df['Amount'] * df['Value']
    df['Amount_Value_Difference'] = df['Amount'] - df['Value']
    
    # 2. Log transformations (handle skewness)
    df['LogAmount'] = np.log1p(np.abs(df['Amount']))
    df['LogValue'] = np.log1p(np.abs(df['Value']))
    
    # 3. Time-based features
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    df['IsLateNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    return df

# Apply feature engineering
train = engineer_features(train)

# ============================================================================
# FEATURE SELECTION (matching notebook)
# ============================================================================
# ID columns to exclude from modeling
id_cols = ['TransactionId', 'BatchId', 'SubscriptionId', 'CustomerId', 
           'CurrencyCode', 'CountryCode', 'TransactionStartTime', 'Date',
           'AccountId', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId',
           'Hour', 'Day', 'Month', 'Weekday']

# Select numerical feature columns (matching the notebook's selection)
feature_cols = ['Amount', 'Value', 'PricingStrategy',
                'Amount_Value_Ratio', 'Amount_Value_Interaction', 'Amount_Value_Difference',
                'LogAmount', 'LogValue', 'IsWeekend', 'IsBusinessHour', 'IsLateNight']

# Prepare X and y
X = train[feature_cols].fillna(0)
y = train['FraudResult']

print(f"Features selected: {feature_cols}")

# ============================================================================
# TRAIN LIGHTGBM MODEL
# ============================================================================
print("Training LightGBM model...")
neg = (y == 0).sum()
pos = (y == 1).sum()

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Create DataFrame for plotting
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

print("\nFeature Importance Ranking:")
print(feature_importance_df.sort_values('Importance', ascending=False).to_string(index=False))

# Create the plot
plt.figure(figsize=(10, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Create horizontal bar chart
bars = plt.barh(feature_importance_df['Feature'], 
                feature_importance_df['Importance'], 
                color='steelblue')

plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance - LightGBM', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('figs/feature_importance.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✓ Feature importance plot saved to figs/feature_importance.png")

