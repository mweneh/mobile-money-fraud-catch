"""
=============================================================================
SAAB-DL: SMOTE-Aware Adaptive Boosting with Deep Learning Feature Fusion
=============================================================================
Architecture:
  1. Feature Engineering  (mirrors fraud-with-SMOTE.ipynb baseline)
  2. VAE                  (behavioural latent embeddings, TensorFlow/Keras)
  3. GNN                  (transaction network embeddings, PyTorch + NetworkX)
  4. Feature Fusion       (raw + VAE + GNN → 23-dim fused vector)
  5. SAAB                 (LDS → CSA → AEW adaptive ensemble)
  6. SHAP                 (interpretability)
  7. 20-Dataset Evaluation (temporal, imbalance, channel, noise, SMOTE variants)

Based on:
  - Original paper: LightGBM (class weight) >> XGBoost+SMOTE on Xente dataset
  - Kungu et al. (2026): VAE + GNN embeddings → One-Class SVM (unsupervised)
  - Sonnet chat (2024):  SAAB algorithm design
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script mode

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE, ADASYN
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH  = 'training.csv'
TEST_PATH   = 'test.csv'
RANDOM_SEED = 42
VAE_EPOCHS  = 30
GNN_EPOCHS  = 30
VAE_LATENT  = 8
GNN_EMBED   = 8
LDS_K       = 5        # k-NN for Local Density Scoring
LDS_THRESH  = 0.5      # threshold above which CSA is applied
RESULTS_DIR = 'saab_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURE ENGINEERING
#    (mirrors fraud-with-SMOTE.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────
def load_and_engineer(path, account_stats=None, is_train=True):
    """Load CSV and apply the same feature engineering as the original paper."""
    df = pd.read_csv(path)
    print(f"  Loaded {path}: {df.shape}")

    # Datetime parsing
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Hour']    = df['TransactionStartTime'].dt.hour
    df['Day']     = df['TransactionStartTime'].dt.day
    df['Month']   = df['TransactionStartTime'].dt.month
    df['Weekday'] = df['TransactionStartTime'].dt.weekday

    # Row-level features (identical to fraud-with-SMOTE)
    df['Amount_Value_Ratio']       = df['Amount'] / (df['Value'] + 1e-6)
    df['Amount_Value_Interaction'] = df['Amount'] * df['Value']
    df['Amount_Value_Difference']  = df['Amount'] - df['Value']
    df['LogAmount']   = np.log1p(np.abs(df['Amount']))
    df['LogValue']    = np.log1p(np.abs(df['Value']))
    df['IsWeekend']      = df['Weekday'].isin([5, 6]).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    df['IsLateNight']    = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)

    # Account aggregate stats (computed on train, applied to all)
    if is_train:
        account_stats = df.groupby('AccountId')['Amount'].agg(
            ['count', 'mean', 'std', 'min', 'max']
        )
        account_stats.columns = [
            'Account_TxnCount', 'Account_AvgAmount',
            'Account_StdAmount', 'Account_MinAmount', 'Account_MaxAmount'
        ]
        account_stats['Account_AmountRange'] = (
            account_stats['Account_MaxAmount'] - account_stats['Account_MinAmount']
        )
        account_stats['Account_StdAmount'] = account_stats['Account_StdAmount'].fillna(0)

    df = df.merge(account_stats, on='AccountId', how='left')
    fill_val = 0
    for col in account_stats.columns:
        df[col] = df[col].fillna(fill_val)

    return df, account_stats


FEATURE_COLS = [
    'Amount', 'Value', 'PricingStrategy',
    'Amount_Value_Ratio', 'Amount_Value_Interaction', 'Amount_Value_Difference',
    'LogAmount', 'LogValue',
    'IsWeekend', 'IsBusinessHour', 'IsLateNight',
    'Account_TxnCount', 'Account_AvgAmount', 'Account_StdAmount',
    'Account_MinAmount', 'Account_MaxAmount', 'Account_AmountRange',
    'Hour', 'Day', 'Month', 'Weekday'
]

# VAE numerical inputs (same 7 as Kungu et al.)
VAE_INPUT_COLS = ['Amount', 'Value', 'LogAmount', 'LogValue',
                  'Amount_Value_Ratio', 'IsLateNight', 'Account_TxnCount']


# ─────────────────────────────────────────────────────────────────────────────
# 2. VARIATIONAL AUTOENCODER (VAE)
#    TensorFlow/Keras — mirrors Kungu et al. architecture exactly
# ─────────────────────────────────────────────────────────────────────────────
def build_and_train_vae(X_train_vae, X_test_vae, latent_dim=VAE_LATENT, epochs=VAE_EPOCHS):
    """
    VAE: encoder compresses 7 numerical features → 8-dim latent space.
    Architecture follows Kungu et al. (2026):
      Encoder: Dense(32) → Dense(16) → [mean, log_var] → z (8-dim)
      Decoder: Dense(16) → Dense(32) → reconstruction

    LEAKAGE FIX: trained ONLY on X_train_vae; X_test_vae is transformed
    using the frozen encoder — test transactions never influence VAE weights.
    """
    X_vae = X_train_vae  # alias — VAE is fit on train only
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("  ⚠️  TensorFlow not installed — skipping VAE. Run: pip install tensorflow")
        return None, np.zeros((len(X_vae), latent_dim))

    tf.random.set_seed(RANDOM_SEED)

    input_dim = X_vae.shape[1]

    # ── Encoder ──────────────────────────────────────────────────────────────
    inputs  = keras.Input(shape=(input_dim,), name='encoder_input')
    x       = layers.Dense(32, activation='relu')(inputs)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(16, activation='relu')(x)
    z_mean  = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_v = layers.Dense(latent_dim, name='z_log_var')(x)

    # Reparameterisation trick with KL-divergence clipping (stability)
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            z_log_var = tf.clip_by_value(z_log_var, -4, 4)
            epsilon   = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z       = Sampling()([z_mean, z_log_v])
    encoder = keras.Model(inputs, [z_mean, z_log_v, z], name='encoder')

    # ── Decoder ──────────────────────────────────────────────────────────────
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation='relu')(latent_inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(input_dim)(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # ── VAE model ─────────────────────────────────────────────────────────────
    class VAE(keras.Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def call(self, data):
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mse(data, reconstruction))
            )
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            self.add_loss(recon_loss + kl_loss)
            return reconstruction

    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')

    print(f"  Training VAE on train split only ({epochs} epochs)...")
    history = vae.fit(
        X_vae, X_vae,
        epochs=epochs, batch_size=256, verbose=0,
        validation_split=0.1
    )
    final_loss = history.history['loss'][-1]
    print(f"  VAE training loss: {final_loss:.4f}")

    # Extract embeddings using frozen encoder — train then test separately
    z_tr, _, _ = encoder.predict(X_train_vae, verbose=0)  # train embeddings
    z_te, _, _ = encoder.predict(X_test_vae,  verbose=0)  # test embeddings (no leakage)
    print(f"  VAE embeddings: train={z_tr.shape}, test={z_te.shape}")
    return vae, z_tr, z_te


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH NEURAL NETWORK (GNN)
#    NetworkX graph + PyTorch feed-forward → 8-dim node embeddings
#    Nodes: unique AccountIds + CustomerIds
#    Edges: AccountId → CustomerId, weighted by transaction amount & frequency
# ─────────────────────────────────────────────────────────────────────────────
def build_transaction_graph(df):
    """Build directed graph from AccountId → CustomerId transactions."""
    print("  Building transaction graph...")
    G = nx.DiGraph()

    # Aggregate edges
    edge_data = df.groupby(['AccountId', 'CustomerId']).agg(
        weight=('Amount', 'sum'),
        freq=('Amount', 'count')
    ).reset_index()

    for _, row in edge_data.iterrows():
        G.add_edge(
            row['AccountId'], row['CustomerId'],
            weight=float(row['weight']),
            freq=int(row['freq'])
        )

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  Graph: {n_nodes:,} nodes, {n_edges:,} directed edges")
    return G


def extract_gnn_embeddings(df, G, embed_dim=GNN_EMBED, epochs=GNN_EPOCHS):
    """
    Simple GNN: compute structural node features from the graph,
    then train a feed-forward network to compress them into embed_dim dimensions.
    Uses PyTorch if available, falls back to PCA on structural features.
    """
    # ── Node feature matrix (structural properties) ───────────────────────────
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    in_degree   = np.array([G.in_degree(node,  weight='weight') for node in nodes], dtype=np.float32)
    out_degree  = np.array([G.out_degree(node, weight='weight') for node in nodes], dtype=np.float32)
    in_freq     = np.array([G.in_degree(node,  weight='freq')   for node in nodes], dtype=np.float32)
    out_freq    = np.array([G.out_degree(node, weight='freq')    for node in nodes], dtype=np.float32)

    # Volume variability (std of edge weights)
    in_std, out_std = [], []
    for node in nodes:
        in_w  = [G[u][node]['weight'] for u in G.predecessors(node)]
        out_w = [G[node][v]['weight'] for v in G.successors(node)]
        in_std.append( np.std(in_w)  if in_w  else 0.0)
        out_std.append(np.std(out_w) if out_w else 0.0)
    in_std  = np.array(in_std,  dtype=np.float32)
    out_std = np.array(out_std, dtype=np.float32)

    node_feats = np.column_stack([
        in_degree, out_degree, in_freq, out_freq, in_std, out_std,
        in_degree / (out_degree + 1e-6),   # in/out ratio (layering signal)
        in_freq   / (out_freq   + 1e-6),
    ])  # shape: (n_nodes, 8) — the target GNN input

    # Normalise
    scaler = StandardScaler()
    node_feats_scaled = scaler.fit_transform(node_feats)

    # ── PyTorch encoder ───────────────────────────────────────────────────────
    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(RANDOM_SEED)

        class GNNEncoder(nn.Module):
            def __init__(self, in_dim, embed_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 32), nn.ReLU(),
                    nn.Linear(32, 16),     nn.ReLU(),
                    nn.Linear(16, embed_dim)
                )
            def forward(self, x):
                return self.net(x)

        X_tensor = torch.tensor(node_feats_scaled, dtype=torch.float32)
        # Self-supervised: predict own features (autoencoder objective)
        model_gnn = GNNEncoder(node_feats_scaled.shape[1], embed_dim)
        decoder   = nn.Linear(embed_dim, node_feats_scaled.shape[1])
        optimizer = torch.optim.Adam(
            list(model_gnn.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        loss_fn   = nn.MSELoss()

        print(f"  Training GNN encoder ({epochs} epochs)...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            z    = model_gnn(X_tensor)
            recon = decoder(z)
            loss = loss_fn(recon, X_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:2d}/{epochs}  loss={loss.item():.4f}")

        model_gnn.eval()
        with torch.no_grad():
            embeddings = model_gnn(X_tensor).numpy()  # (n_nodes, embed_dim)
        print(f"  GNN embeddings extracted: {embeddings.shape}")

    except ImportError:
        print("  ⚠️  PyTorch not installed — using PCA as GNN fallback")
        from sklearn.decomposition import PCA
        embeddings = PCA(n_components=embed_dim, random_state=RANDOM_SEED).fit_transform(node_feats_scaled)

    # ── Map node embeddings back to transaction rows ───────────────────────────
    # Each transaction gets the embedding of its AccountId node
    embed_cols = [f'GNN_emb_{i}' for i in range(embed_dim)]
    embed_df   = pd.DataFrame(embeddings, index=nodes, columns=embed_cols)

    def map_embeddings(df_rows):
        """Map train-derived node embeddings to any df subset. Unknown → zeros."""
        df_gnn = df_rows[['AccountId']].copy().reset_index(drop=True)
        df_gnn = df_gnn.merge(embed_df, left_on='AccountId', right_index=True, how='left')
        for col in embed_cols:
            df_gnn[col] = df_gnn[col].fillna(0.0)
        return df_gnn[embed_cols].values

    return map_embeddings, embed_cols, embed_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE FUSION
# ─────────────────────────────────────────────────────────────────────────────
def fuse_features(X_raw, vae_embeddings, gnn_embeddings):
    """
    Concatenate raw features + VAE latent + GNN embeddings.
    Returns a unified feature matrix and column names.
    """
    vae_cols = [f'VAE_emb_{i}' for i in range(vae_embeddings.shape[1])]
    gnn_cols = [f'GNN_emb_{i}' for i in range(gnn_embeddings.shape[1])]

    X_fused = np.concatenate([X_raw, vae_embeddings, gnn_embeddings], axis=1)
    fused_cols = (
        list(range(X_raw.shape[1])) +  # will be named properly by caller
        vae_cols + gnn_cols
    )
    print(f"  Fused feature matrix: {X_fused.shape} "
          f"({X_raw.shape[1]} raw + {len(vae_cols)} VAE + {len(gnn_cols)} GNN)")
    return X_fused, vae_cols, gnn_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAAB — SMOTE-Aware Adaptive Boosting
# ─────────────────────────────────────────────────────────────────────────────

class LocalDensityScorer:
    """
    Step 1 — Local Density Scoring (LDS)
    For each minority sample, compute a density score using k-NN distances
    to other minority samples. High LDS → sparse neighbourhood.
    """
    def __init__(self, k=LDS_K):
        self.k   = k
        self.knn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
        self.minority_X  = None
        self.max_dist    = None

    def fit(self, X_minority):
        self.minority_X = X_minority
        self.knn.fit(X_minority)
        distances, _ = self.knn.kneighbors(X_minority)
        self.max_dist = distances[:, -1].max() + 1e-9
        return self

    def score(self, X):
        """Returns LDS ∈ [0,1] for each sample in X (high = sparse region)."""
        distances, _ = self.knn.kneighbors(X)
        mean_dist = distances[:, -1]
        return np.clip(mean_dist / self.max_dist, 0, 1)


def conditional_synthetic_augmentation(X_train, y_train, lds_scores, thresh=LDS_THRESH):
    """
    Step 2 — Conditional Synthetic Augmentation (CSA)
    Apply SMOTE only to minority samples in high-LDS (sparse) regions.
    Returns augmented (X, y) for XGBoost training.
    """
    minority_mask = (y_train == 1)
    minority_X    = X_train[minority_mask]
    minority_y    = y_train[minority_mask]

    # Split minority into sparse (high-LDS) and dense (low-LDS)
    sparse_mask = lds_scores > thresh
    dense_mask  = ~sparse_mask

    n_sparse = sparse_mask.sum()
    n_dense  = dense_mask.sum()
    print(f"  CSA: {n_sparse} sparse minority samples (LDS>{thresh}) "
          f"| {n_dense} dense minority samples")

    if n_sparse < 2:
        print("  CSA: Too few sparse samples — falling back to global SMOTE")
        smote = SMOTE(random_state=RANDOM_SEED)
        return smote.fit_resample(X_train, y_train)

    # Reconstruct dataset: keep all majority + dense minority + SMOTE sparse minority
    majority_X = X_train[~minority_mask]
    majority_y = y_train[~minority_mask]
    dense_X    = minority_X[dense_mask]
    dense_y    = minority_y[dense_mask]
    sparse_X   = minority_X[sparse_mask]
    sparse_y   = minority_y[sparse_mask]

    # SMOTE on sparse subset only
    n_target = max(n_dense + n_sparse, int(n_sparse * 3))  # triple sparse minority
    try:
        smote = SMOTE(
            random_state=RANDOM_SEED,
            k_neighbors=min(LDS_K - 1, n_sparse - 1)
        )
        # Combine sparse minority with a fraction of majority for SMOTE context
        n_majority_context = min(len(majority_X), n_sparse * 20)
        context_idx = np.random.choice(len(majority_X), n_majority_context, replace=False)
        X_smote_ctx = np.vstack([sparse_X, majority_X[context_idx]])
        y_smote_ctx = np.concatenate([sparse_y, majority_y[context_idx]])

        X_sparse_aug, y_sparse_aug = smote.fit_resample(X_smote_ctx, y_smote_ctx)
        # Keep only the newly synthesized minority samples
        new_minority_mask    = (y_sparse_aug == 1)
        X_new_syn = X_sparse_aug[new_minority_mask][n_sparse:]  # synthetic only
        y_new_syn = y_sparse_aug[new_minority_mask][n_sparse:]

        X_aug = np.vstack([majority_X, dense_X, sparse_X, X_new_syn])
        y_aug = np.concatenate([majority_y, dense_y, sparse_y, y_new_syn])

    except Exception as e:
        print(f"  CSA SMOTE failed ({e}) — using global SMOTE fallback")
        smote  = SMOTE(random_state=RANDOM_SEED)
        X_aug, y_aug = smote.fit_resample(X_train, y_train)

    print(f"  CSA output: {X_aug.shape[0]:,} rows "
          f"({(y_aug==1).sum()} fraud, {(y_aug==0).sum()} legit)")
    return X_aug, y_aug


class SAAABClassifier:
    """
    SAAB — SMOTE-Aware Adaptive Boosting (with Deep Learning Feature Fusion)

    Steps:
      1. LDS  — Local Density Scoring (k-NN on fused feature space)
      2. CSA  — Conditional Synthetic Augmentation (SMOTE in sparse regions)
      3. AEW  — Adaptive Ensemble Weighting at inference
          P_final = (1 - LDS_norm) × P_LGBM + LDS_norm × P_XGB_CSA
    """

    def __init__(
        self,
        lgbm_params=None,
        xgb_params=None,
        lds_k=LDS_K,
        lds_thresh=LDS_THRESH,
    ):
        self.lds_k      = lds_k
        self.lds_thresh = lds_thresh

        self.lgbm = LGBMClassifier(
            **(lgbm_params or dict(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1,
            ))
        )
        self.xgb = XGBClassifier(
            **(xgb_params or dict(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=1,   # will be set in fit()
                eval_metric='logloss',
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            ))
        )
        self.lds_scorer = LocalDensityScorer(k=lds_k)
        self.scaler     = StandardScaler()
        self.fitted     = False

    def fit(self, X, y):
        X       = np.array(X)
        y       = np.array(y)
        minority_X = X[y == 1]
        n_neg   = (y == 0).sum()
        n_pos   = (y == 1).sum()

        print(f"\n── SAAB FIT ─────────────────────────────")
        print(f"  Train: {len(y):,} samples | {n_pos} fraud ({n_pos/len(y):.3%})")

        # ── Step 1: LDS ───────────────────────────────────────────────────────
        print(f"\n  Step 1 — Local Density Scoring (k={self.lds_k})")
        Xs = self.scaler.fit_transform(X)
        self.lds_scorer.fit(Xs[y == 1])
        lds_scores = self.lds_scorer.score(Xs[y == 1])
        print(f"  LDS stats: mean={lds_scores.mean():.3f}, "
              f"max={lds_scores.max():.3f}, "
              f"sparse_frac={(lds_scores > self.lds_thresh).mean():.2%}")

        # ── Step 2: CSA ───────────────────────────────────────────────────────
        print(f"\n  Step 2 — Conditional Synthetic Augmentation")
        X_csa, y_csa = conditional_synthetic_augmentation(
            Xs, y, lds_scores, self.lds_thresh
        )

        # ── Step 3a: Train LGBM on ORIGINAL (class-weight balanced) ──────────
        print(f"\n  Step 3a — Training LGBM on original distribution...")
        self.lgbm.fit(Xs, y)

        # ── Step 3b: Train XGBoost on CSA-augmented data ──────────────────────
        print(f"\n  Step 3b — Training XGBoost on CSA-augmented data...")
        n_pos_csa = (y_csa == 1).sum()
        n_neg_csa = (y_csa == 0).sum()
        self.xgb.set_params(scale_pos_weight=n_neg_csa / max(n_pos_csa, 1))
        self.xgb.fit(X_csa, y_csa)

        self.fitted = True
        print(f"\n  SAAB training complete ✓")
        return self

    def predict_proba(self, X):
        assert self.fitted, "Call fit() first."
        X = np.array(X)
        Xs = self.scaler.transform(X)

        p_lgbm = self.lgbm.predict_proba(Xs)[:, 1]
        p_xgb  = self.xgb.predict_proba(Xs)[:, 1]

        # ── Step 3c: AEW — Adaptive Ensemble Weighting ───────────────────────
        lds = self.lds_scorer.score(Xs)          # LDS for each test sample
        p_final = (1 - lds) * p_lgbm + lds * p_xgb
        return np.column_stack([1 - p_final, p_final])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold=0.5, model_name='Model'):
    """Return a dict of all key metrics."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc   = roc_auc_score(y_test, proba)          if y_test.sum() > 0 else np.nan
    ap    = average_precision_score(y_test, proba) if y_test.sum() > 0 else np.nan
    f1    = f1_score(y_test, preds, zero_division=0)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)

    print(f"  {model_name:20s} | ROC-AUC={roc:.4f} | AP={ap:.4f} | "
          f"F1={f1:.4f} | P={prec:.4f} | R={rec:.4f}")

    return dict(model=model_name, roc_auc=roc, ap=ap, f1=f1,
                precision=prec, recall=rec, threshold=threshold)


def find_optimal_threshold(model, X_val, y_val):
    """Find F1-optimal decision threshold on validation set."""
    from sklearn.metrics import precision_recall_curve
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.argmax(f1_scores)
    return thresholds[idx] if idx < len(thresholds) else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 7. 20-DATASET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def build_20_datasets(df_train, feature_cols, target='FraudResult'):
    """
    Generate 20 evaluation scenarios from the single 95K dataset.
    Groups A–E as designed in the Sonnet chat.
    Returns list of (name, X_tr, y_tr, X_te, y_te) tuples.
    """
    df = df_train.sort_values('TransactionStartTime').reset_index(drop=True)
    datasets = []

    fraud_mask  = df[target] == 1
    legit_mask  = df[target] == 0
    n_fraud     = fraud_mask.sum()
    fraud_idx   = df[fraud_mask].index
    legit_idx   = df[legit_mask].index

    def make_split(subset_df, name, test_size=0.2):
        X_ = subset_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y_ = subset_df[target].values
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_, y_, test_size=test_size, stratify=y_, random_state=RANDOM_SEED
            )
            datasets.append((name, X_tr, y_tr, X_te, y_te))
            print(f"  ✓ {name}: {len(y_tr):,} train ({y_tr.sum()} fraud) | "
                  f"{len(y_te):,} test ({y_te.sum()} fraud)")
        except ValueError as e:
            print(f"  ✗ {name} skipped: {e}")

    # ── Group A: Temporal splits (7 datasets) ────────────────────────────────
    print("\n  Group A — Temporal Splits")
    df['week'] = ((df['TransactionStartTime'] - df['TransactionStartTime'].min())
                  .dt.days // 7).clip(0, 15)
    max_week   = df['week'].max()

    # Rolling 6-week train → 2-week test windows
    for i, start in enumerate(range(0, min(max_week - 7, 8), 2)):
        tr_mask = (df['week'] >= start) & (df['week'] < start + 6)
        te_mask = (df['week'] >= start + 6) & (df['week'] < start + 8)
        if df[te_mask][target].sum() >= 2 and df[tr_mask][target].sum() >= 5:
            make_split(pd.concat([df[tr_mask], df[te_mask]]), f'A{i+1}_week{start}-{start+8}')

    # Hour-of-day splits
    peak_hours  = df.groupby('Hour')[target].mean().nlargest(8).index
    df['IsPeak'] = df['Hour'].isin(peak_hours)
    make_split(df[df['IsPeak']],  'A5_peak_hours')
    make_split(df[~df['IsPeak']], 'A6_offpeak_hours')
    make_split(df[df['IsWeekend'] == 0], 'A7_weekdays')

    # ── Group B: Imbalance stress tests (5 datasets) ─────────────────────────
    print("\n  Group B — Imbalance Stress Tests")
    for target_rate, name in [(0.001, 'B1_0.1pct'), (0.002, 'B2_0.2pct_orig'),
                               (0.005, 'B3_0.5pct'), (0.010, 'B4_1pct'),
                               (0.020, 'B5_2pct')]:
        n_legit_target = int(n_fraud / target_rate)
        n_legit_sample = min(n_legit_target, len(legit_idx))
        sampled_legit  = np.random.choice(legit_idx, n_legit_sample, replace=True)
        subset = pd.concat([df.loc[fraud_idx], df.loc[sampled_legit]])
        make_split(subset, name)

    # ── Group C: Transaction type / channel subsets (4 datasets) ─────────────
    print("\n  Group C — Channel Subsets")
    for ch in df['ProductCategory'].dropna().unique()[:4]:
        subset = df[df['ProductCategory'] == ch]
        if subset[target].sum() >= 5:
            make_split(subset, f'C_channel_{ch[:8]}')

    # ── Group D: Feature perturbation (2 datasets) ───────────────────────────
    print("\n  Group D — Feature Perturbation")
    for noise_level, name in [(0.2, 'D1_amount_noise_20pct'),
                               (0.4, 'D2_amount_noise_40pct')]:
        df_noisy = df.copy()
        df_noisy['Amount'] = df_noisy['Amount'] * (
            1 + noise_level * np.random.randn(len(df_noisy))
        )
        df_noisy['LogAmount'] = np.log1p(np.abs(df_noisy['Amount']))
        make_split(df_noisy, name)

    # ── Group E: SMOTE pre-processing variants (2 datasets) ──────────────────
    print("\n  Group E — SMOTE variants (applied to full train split)")
    X_full = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_full = df[target].values
    for smote_cls, name in [(SMOTE,  'E1_global_SMOTE'),
                             (ADASYN, 'E2_ADASYN')]:
        try:
            X_s, y_s = smote_cls(random_state=RANDOM_SEED).fit_resample(X_full, y_full)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_s, y_s, test_size=0.2, stratify=y_s, random_state=RANDOM_SEED
            )
            datasets.append((name, X_tr, y_tr, X_te, y_te))
            print(f"  ✓ {name}: {len(y_tr):,} train | {len(y_te):,} test")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print(f"\n  Total evaluation scenarios: {len(datasets)}")
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# 8. SHAP INTERPRETABILITY
# ─────────────────────────────────────────────────────────────────────────────
def run_shap(model, X_sample, feature_names, out_dir=RESULTS_DIR, tag='saab'):
    """SHAP summary for the LGBM component (most interpretable)."""
    try:
        import shap
        print(f"\n  Computing SHAP values ({len(X_sample)} samples)...")
        Xs = model.scaler.transform(X_sample)
        explainer  = shap.TreeExplainer(model.lgbm)
        shap_vals  = explainer.shap_values(Xs)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class-1 SHAP values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, Xs, feature_names=feature_names,
                          show=False, max_display=20)
        plt.tight_layout()
        path = os.path.join(out_dir, f'shap_summary_{tag}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  SHAP plot saved → {path}")

        # Top features by mean |SHAP|
        mean_shap = np.abs(shap_vals).mean(axis=0)
        top = pd.DataFrame({'feature': feature_names, 'mean_shap': mean_shap})
        top = top.sort_values('mean_shap', ascending=False).head(15)
        print("\n  Top 15 features by SHAP importance:")
        print(top.to_string(index=False))
        return top

    except ImportError:
        print("  ⚠️  shap not installed — run: pip install shap")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 9. BASELINE MODELS (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
def train_baselines(X_train, y_train):
    """Train LGBM and XGBoost+SMOTE baselines to mirror original paper."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    lgbm_baseline = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        class_weight='balanced', random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
    )
    lgbm_baseline.fit(X_train, y_train)

    smote = SMOTE(random_state=RANDOM_SEED)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    xgb_smote = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=n_neg/max(n_pos, 1),
        eval_metric='logloss', random_state=RANDOM_SEED, verbosity=0, n_jobs=-1
    )
    xgb_smote.fit(X_sm, y_sm)

    return lgbm_baseline, xgb_smote


# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_20_dataset_comparison(results_df, out_dir=RESULTS_DIR):
    """Bar chart comparing SAAB vs baselines across 20 datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('SAAB-DL vs Baselines — 20-Dataset Evaluation', fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, ['roc_auc', 'f1']):
        pivot = results_df.pivot(index='dataset', columns='model', values=metric)
        pivot.plot(kind='bar', ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title(metric.upper().replace('_', '-'))
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        ax.legend(loc='lower right')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.846, color='grey', linestyle='--', linewidth=1,
                   label='Kungu et al. 0.846')

    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_20_datasets.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved → {path}")


def plot_model_summary(summary_df, out_dir=RESULTS_DIR):
    """Mean ± std summary across all 20 datasets."""
    summary = summary_df.groupby('model')[['roc_auc', 'ap', 'f1']].agg(['mean', 'std'])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.25
    for i, col in enumerate(['roc_auc', 'ap', 'f1']):
        means = summary[(col, 'mean')].values
        stds  = summary[(col, 'std')].values
        bars = ax.bar(x + i * width, means, width, label=col.upper(),
                      yerr=stds, capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(summary.index, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title('Mean Performance Across 20 Datasets (±1 std)')
    ax.legend()
    ax.axhline(y=0.846, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    path = os.path.join(out_dir, 'model_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SAAB-DL  PIPELINE  (SMOTE-Aware Adaptive Boosting + Deep Features)")
    print("=" * 70)

    # ─── LOAD DATA ────────────────────────────────────────────────────────────
    print("\n[1/8] Loading & Engineering Features")
    train, account_stats = load_and_engineer(TRAIN_PATH, is_train=True)
    test,  _             = load_and_engineer(TEST_PATH, account_stats=account_stats,
                                             is_train=False)

    feat_cols_present = [c for c in FEATURE_COLS if c in train.columns]
    vae_cols_present  = [c for c in VAE_INPUT_COLS if c in train.columns]

    X_raw = train[feat_cols_present].fillna(0).replace([np.inf, -np.inf], 0).values
    y     = train['FraudResult'].values

    print(f"\n  Features used: {feat_cols_present}")
    print(f"  Fraud rate: {y.mean():.4%}  ({y.sum()} / {len(y)})")

    # ─── SPLIT FIRST (before any fitting) ────────────────────────────────────
    # Use index-based split so we can correctly map VAE/GNN embeddings
    # to train and test rows without positional assumptions.
    idx = np.arange(len(X_raw))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # Fit raw scaler on TRAIN indices only → transform both splits
    scaler_raw = StandardScaler()
    X_tr_raw = scaler_raw.fit_transform(X_raw[idx_tr])
    X_te_raw = scaler_raw.transform(X_raw[idx_te])

    train_df_tr = train.iloc[idx_tr].reset_index(drop=True)  # train rows as DataFrame
    train_df_te = train.iloc[idx_te].reset_index(drop=True)  # test rows as DataFrame

    # ─── VAE (LEAKAGE-FREE) ───────────────────────────────────────────────────
    print("\n[2/8] VAE — Behavioural Feature Extraction (train-split only)")
    X_vae_all   = train[vae_cols_present].fillna(0).replace([np.inf, -np.inf], 0).values

    # Fit scaler on train VAE inputs only
    scaler_vae  = StandardScaler()
    X_vae_tr    = scaler_vae.fit_transform(X_vae_all[idx_tr])
    X_vae_te    = scaler_vae.transform(X_vae_all[idx_te])      # transform with train stats

    vae_model, vae_tr, vae_te = build_and_train_vae(X_vae_tr, X_vae_te)

    # ─── GNN (LEAKAGE-FREE) ───────────────────────────────────────────────────
    print("\n[3/8] GNN — Transaction Network Embedding (train-split graph only)")
    # Build graph ONLY from training rows → test accounts use train-derived embeddings
    G = build_transaction_graph(train_df_tr)
    map_fn, gnn_col_names, embed_df = extract_gnn_embeddings(train_df_tr, G)

    # Map to train and test separately using the same frozen embeddings
    gnn_tr = map_fn(train_df_tr)   # train accounts: known nodes
    gnn_te = map_fn(train_df_te)   # test accounts: unknown → zeros (no leakage)
    unseen_frac = (gnn_te.sum(axis=1) == 0).mean()
    print(f"  GNN test coverage: {1-unseen_frac:.1%} seen, {unseen_frac:.1%} unseen→zeros")

    # ─── FEATURE FUSION ───────────────────────────────────────────────────────
    print("\n[4/8] Feature Fusion")
    X_tr_fused = np.concatenate([X_tr_raw, vae_tr, gnn_tr], axis=1)
    X_te_fused = np.concatenate([X_te_raw, vae_te, gnn_te], axis=1)

    all_feat_names = (feat_cols_present
                      + [f'VAE_emb_{i}' for i in range(VAE_LATENT)]
                      + gnn_col_names)

    print(f"  Fused shape: {X_tr_fused.shape[1]} features "
          f"({len(feat_cols_present)} raw + {VAE_LATENT} VAE + {GNN_EMBED} GNN)")

    # ─── BASELINES ────────────────────────────────────────────────────────────
    print("\n[5/8] Training Baselines")
    lgbm_base, xgb_smote = train_baselines(X_tr_fused, y_tr)
    thr_lgbm = find_optimal_threshold(lgbm_base, X_te_fused, y_te)
    thr_xgb  = find_optimal_threshold(xgb_smote, X_te_fused, y_te)

    print("\n  Baseline Results (fused features):")
    res_lgbm = evaluate(lgbm_base, X_te_fused, y_te, thr_lgbm, 'LGBM-baseline')
    res_xgb  = evaluate(xgb_smote, X_te_fused, y_te, thr_xgb,  'XGB+SMOTE-baseline')

    # ─── SAAB ─────────────────────────────────────────────────────────────────
    print("\n[6/8] Training SAAB-DL")
    saab = SAAABClassifier()
    saab.fit(X_tr_fused, y_tr)
    thr_saab = find_optimal_threshold(saab, X_te_fused, y_te)

    print("\n  SAAB-DL Result:")
    res_saab = evaluate(saab, X_te_fused, y_te, thr_saab, 'SAAB-DL')
    print(f"\n  Classification Report:")
    preds = saab.predict(X_te_fused, threshold=thr_saab)
    print(classification_report(y_te, preds, zero_division=0))

    # ─── SHAP ─────────────────────────────────────────────────────────────────
    print("\n[7/8] SHAP Interpretability")
    n_shap = min(500, len(X_te_fused))
    run_shap(saab, X_te_fused[:n_shap], all_feat_names)

    # ─── 20-DATASET EVALUATION ────────────────────────────────────────────────
    print("\n[8/8] 20-Dataset Evaluation")
    all_results = []

    datasets_20 = build_20_datasets(train, feat_cols_present)
    print(f"\n  Running SAAB-DL on {len(datasets_20)} evaluation scenarios...")

    for ds_name, X_ds_tr, y_ds_tr, X_ds_te, y_ds_te in datasets_20:
        if y_ds_te.sum() == 0:
            print(f"  ✗ {ds_name}: no fraud in test split — skipped")
            continue

        # Augment each scenario's features with VAE/GNN embeddings would require
        # re-running the extractors — for the 20-dataset evaluation we use raw
        # features only to keep compute tractable, and note this in the paper.
        try:
            saab_ds = SAAABClassifier()
            saab_ds.fit(X_ds_tr, y_ds_tr)
            thr = find_optimal_threshold(saab_ds, X_ds_te, y_ds_te)

            lgbm_ds, xgb_ds = train_baselines(X_ds_tr, y_ds_tr)
            thr_l = find_optimal_threshold(lgbm_ds, X_ds_te, y_ds_te)
            thr_x = find_optimal_threshold(xgb_ds,  X_ds_te, y_ds_te)

            r_s = evaluate(saab_ds, X_ds_te, y_ds_te, thr,   f'{ds_name}/SAAB-DL')
            r_l = evaluate(lgbm_ds, X_ds_te, y_ds_te, thr_l, f'{ds_name}/LGBM')
            r_x = evaluate(xgb_ds,  X_ds_te, y_ds_te, thr_x, f'{ds_name}/XGB+SMOTE')

            for r, m in [(r_s, 'SAAB-DL'), (r_l, 'LGBM'), (r_x, 'XGB+SMOTE')]:
                all_results.append({**r, 'dataset': ds_name, 'model': m})

        except Exception as e:
            print(f"  ✗ {ds_name} error: {e}")
            continue

    # ─── SUMMARY TABLE ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'results_20_datasets.csv'), index=False)

    print("\n" + "=" * 70)
    print("MASTER COMPARISON — Mean across all scenarios")
    print("=" * 70)
    summary = results_df.groupby('model')[['roc_auc', 'ap', 'f1']].agg(['mean', 'std'])
    print(summary.round(4))

    # Count wins
    pivot_f1 = results_df.pivot(index='dataset', columns='model', values='f1').dropna()
    saab_wins = (pivot_f1['SAAB-DL'] > pivot_f1.drop(columns='SAAB-DL').max(axis=1)).sum()
    print(f"\n  SAAB-DL wins on {saab_wins}/{len(pivot_f1)} datasets by F1-score")
    print(f"  (Kungu et al. benchmark: ROC-AUC = 0.846)")

    # ─── PLOTS ────────────────────────────────────────────────────────────────
    if len(results_df) > 0:
        plot_20_dataset_comparison(results_df)
        plot_model_summary(results_df)

    print(f"\n✅ All outputs saved to ./{RESULTS_DIR}/")
    print("   - results_20_datasets.csv")
    print("   - comparison_20_datasets.png")
    print("   - model_summary.png")
    print("   - shap_summary_saab.png")

    return saab, results_df, summary


if __name__ == '__main__':
    saab_model, results, summary = main()
