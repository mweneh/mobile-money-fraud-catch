import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    roc_auc_score,
    precision_recall_curve,
    auc
)
from lightgbm import LGBMClassifier


def engineer_features(df, is_train=True):
    """Apply advanced feature engineering"""
    print("  Engineering features...")
    
    # 1. Amount and Value interactions
    df['Amount_Value_Ratio'] = df['Amount'] / (df['Value'] + 1e-6)
    df['Amount_Value_Interaction'] = df['Amount'] * df['Value']
    df['Amount_Value_Difference'] = df['Amount'] - df['Value']
    
    # 2. Log transformations
    df['LogAmount'] = np.log1p(np.abs(df['Amount']))
    df['LogValue'] = np.log1p(np.abs(df['Value']))
    
    # 3. Time-based features
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    df['IsLateNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    # 4. Account-level aggregates (only on training data to avoid leakage)
    if is_train and 'AccountId' in df.columns:
        account_stats = df.groupby('AccountId')['Amount'].agg(['count', 'mean', 'std', 'min', 'max'])
        account_stats.columns = ['Account_TxnCount', 'Account_AvgAmount', 
                                 'Account_StdAmount', 'Account_MinAmount', 'Account_MaxAmount']
        account_stats['Account_AmountRange'] = account_stats['Account_MaxAmount'] - account_stats['Account_MinAmount']
        account_stats['Account_StdAmount'] = account_stats['Account_StdAmount'].fillna(0)
        
        df = df.merge(account_stats, left_on='AccountId', right_index=True, how='left')
    
    return df


def run_pipeline():
    print("="*70)
    print("  ENHANCED FRAUD DETECTION PIPELINE")
    print("="*70 + "\n")
    
    print("📂 Loading data...")
    train_df = pd.read_csv('training.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"  Training: {train_df.shape}")
    print(f"  Test: {test_df.shape}\n")

    print("🔧 Preprocessing and feature engineering...")
    # Date conversion
    train_df['TransactionStartTime'] = pd.to_datetime(train_df['TransactionStartTime'])
    test_df['TransactionStartTime'] = pd.to_datetime(test_df['TransactionStartTime'])

    # Basic time features
    for df in [train_df, test_df]:
        df['Hour'] = df['TransactionStartTime'].dt.hour
        df['Day'] = df['TransactionStartTime'].dt.day
        df['Month'] = df['TransactionStartTime'].dt.month
        df['Weekday'] = df['TransactionStartTime'].dt.weekday
    
    # Advanced feature engineering
    train_df = engineer_features(train_df, is_train=True)
    test_df = engineer_features(test_df, is_train=False)
    
    # ID columns to drop
    id_cols = ['TransactionId', 'BatchId', 'SubscriptionId', 'CustomerId', 
               'CurrencyCode', 'CountryCode', 'TransactionStartTime']
    
    # Keep AccountId for account features, but drop it later
    cols_to_drop = [col for col in id_cols if col != 'AccountId']
    
    # Prepare features
    # Only use features that exist in both train and test
    train_feature_cols = [col for col in train_df.columns 
                         if col not in ['FraudResult'] + id_cols 
                         and train_df[col].dtype in [np.number, 'int64', 'float64']]
    
    test_feature_cols = [col for col in test_df.columns 
                        if col not in id_cols 
                        and test_df[col].dtype in [np.number, 'int64', 'float64']]
    
    # Use intersection of features (only those in both datasets)
    feature_cols = [col for col in train_feature_cols if col in test_feature_cols]
    
    X = train_df[feature_cols].fillna(0)
    y = train_df['FraudResult']
    X_test_submit = test_df[feature_cols].fillna(0)
    
    print(f"  ✅ Total features: {len(feature_cols)}\n")

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Define models with improved hyperparameters
    models = {
        'LightGBM': LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            C=0.1
        ),
        'LinearSVC': LinearSVC(
            max_iter=5000,
            class_weight='balanced',
            random_state=42,
            dual=False
        )
    }

    results = {}

    print("="*70)
    print("  MODEL COMPARISON")
    print("="*70 + "\n")

    for name, model in models.items():
        print(f"🤖 Training {name}...")
        
        # Use scaled data for LinearSVC and Logistic Regression
        if name in ['LinearSVC', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            # Get scores
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_val_scaled)
            else:
                y_score = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_score = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_score)
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_val, y_score)
        pr_auc = auc(recall, precision)
        
        results[name] = {
            'F1': f1, 
            'ROC-AUC': roc_auc, 
            'PR-AUC': pr_auc,
            'model': model
        }
        
        print(f"\n--- {name} Results ---")
        print(classification_report(y_val, y_pred, digits=3))
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"F1:      {f1:.4f}")
        print("-" * 70 + "\n")

    # Select best model based on PR-AUC (better for imbalanced data)
    best_model_name = max(results, key=lambda x: results[x]['PR-AUC'])
    
    print("\n" + "="*70)
    print(f"  🏆 BEST MODEL: {best_model_name}")
    print(f"  PR-AUC:  {results[best_model_name]['PR-AUC']:.4f}")
    print(f"  ROC-AUC: {results[best_model_name]['ROC-AUC']:.4f}")
    print(f"  F1:      {results[best_model_name]['F1']:.4f}")
    print("="*70 + "\n")
    
    # Print comparison table
    print("📊 Model Comparison Summary:")
    print("-" * 70)
    print(f"{'Model':<20} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10}")
    print("-" * 70)
    for name, res in sorted(results.items(), key=lambda x: x[1]['PR-AUC'], reverse=True):
        print(f"{name:<20} {res['ROC-AUC']:>10.4f} {res['PR-AUC']:>10.4f} {res['F1']:>10.4f}")
    print("-" * 70 + "\n")

    # Retrain best model on full dataset
    print(f"🔄 Retraining {best_model_name} on full dataset...")
    best_model = results[best_model_name]['model']
    
    if best_model_name in ['LinearSVC', 'Logistic Regression']:
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test_submit)
        best_model.fit(X_scaled, y)
        test_preds = best_model.predict(X_test_scaled)
    else:
        best_model.fit(X, y)
        test_preds = best_model.predict(X_test_submit)
    
    # Generate submission
    submission = pd.DataFrame({
        'TransactionId': test_df['TransactionId'], 
        'FraudResult': test_preds
    })
    submission.to_csv('submission.csv', index=False)
    
    print(f"✅ Submission saved to submission.csv")
    print(f"   Predicted {test_preds.sum()} fraudulent transactions out of {len(test_preds)}")
    
    # Save model and scaler
    print("\n💾 Saving model artifacts...")
    import joblib
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("   Saved model.pkl and scaler.pkl")

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
