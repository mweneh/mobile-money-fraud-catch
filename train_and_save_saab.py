import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from saab_deep_pipeline import (
    load_and_engineer, FEATURE_COLS, VAE_INPUT_COLS, RANDOM_SEED,
    build_and_train_vae, build_transaction_graph, extract_gnn_embeddings,
    SAAABClassifier
)

def build_model_artifacts():
    print("Loading data...")
    train, account_stats = load_and_engineer('training.csv', is_train=True)
    feat_cols_present = [c for c in FEATURE_COLS if c in train.columns]
    vae_cols_present  = [c for c in VAE_INPUT_COLS if c in train.columns]
    
    X_raw = train[feat_cols_present].fillna(0).replace([np.inf, -np.inf], 0).values
    y     = train['FraudResult'].values
    
    # Only training data should dictate scalers and encoders. We use 80% train.
    idx = np.arange(len(X_raw))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    print("Scaling raw features...")
    scaler_raw = StandardScaler()
    X_tr_raw = scaler_raw.fit_transform(X_raw[idx_tr])
    
    print("Training VAE...")
    X_vae_all   = train[vae_cols_present].fillna(0).replace([np.inf, -np.inf], 0).values
    scaler_vae  = StandardScaler()
    X_vae_tr    = scaler_vae.fit_transform(X_vae_all[idx_tr])
    
    vae_model, vae_tr, _ = build_and_train_vae(X_vae_tr, X_vae_tr, epochs=30)
    
    print("Generating GNN API Embeddings...")
    train_df_tr = train.iloc[idx_tr].reset_index(drop=True)
    G = build_transaction_graph(train_df_tr)
    map_fn, gnn_col_names, embed_df = extract_gnn_embeddings(train_df_tr, G)
    gnn_tr = map_fn(train_df_tr)
    
    print("Fusing features and training SAAB-DL...")
    X_tr_fused = np.concatenate([X_tr_raw, vae_tr, gnn_tr], axis=1)
    
    saab = SAAABClassifier()
    saab.fit(X_tr_fused, y_tr)
    
    print("Saving artifacts...")
    # Clean the SAAAB model slightly to ensure it unpickles okay
    joblib.dump(saab, 'model.pkl')
    joblib.dump(scaler_raw, 'scaler_raw.pkl')
    joblib.dump(scaler_vae, 'scaler_vae.pkl')
    
    # Save VAE encoder
    vae_model.encoder.save('vae_encoder.keras')
    
    # Save GNN embeddings
    embed_df.to_pickle('gnn_embeddings.pkl')
    
    # Save stats and columns
    account_stats.to_csv('account_stats_artifact.csv')
    with open('feature_cols.json', 'w') as f:
        json.dump(feat_cols_present, f)
    with open('vae_cols.json', 'w') as f:
        json.dump(vae_cols_present, f)
        
    print("Completed! The pkl and relevant artifacts are now updated with the SAAB-DL model.")

if __name__ == '__main__':
    build_model_artifacts()
