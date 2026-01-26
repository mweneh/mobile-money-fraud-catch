import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
from lightgbm import LGBMClassifier


FIG_DIR = Path("figs")
FIG_DIR.mkdir(exist_ok=True)


def engineer_features(df: pd.DataFrame, *, with_account_aggregates: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Datetime parsing
    if "TransactionStartTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
        df["Hour"] = df["TransactionStartTime"].dt.hour
        df["Day"] = df["TransactionStartTime"].dt.day
        df["Month"] = df["TransactionStartTime"].dt.month
        df["Weekday"] = df["TransactionStartTime"].dt.weekday

    # Amount/Value interaction features
    if "Amount" in df.columns and "Value" in df.columns:
        df["Amount_Value_Ratio"] = df["Amount"] / (df["Value"] + 1e-6)
        df["Amount_Value_Interaction"] = df["Amount"] * df["Value"]
        df["Amount_Value_Difference"] = df["Amount"] - df["Value"]
        df["LogAmount"] = np.log1p(np.abs(df["Amount"]))
        df["LogValue"] = np.log1p(np.abs(df["Value"]))

    # Temporal flags
    if "Weekday" in df.columns:
        df["IsWeekend"] = df["Weekday"].isin([5, 6]).astype(int)
    if "Hour" in df.columns:
        df["IsBusinessHour"] = ((df["Hour"] >= 9) & (df["Hour"] <= 17)).astype(int)
        df["IsLateNight"] = ((df["Hour"] >= 22) | (df["Hour"] <= 6)).astype(int)

    # Account aggregates (computed within available df)
    if with_account_aggregates and "AccountId" in df.columns and "Amount" in df.columns:
        stats = df.groupby("AccountId")["Amount"].agg(["count", "mean", "std", "min", "max"]).rename(
            columns={
                "count": "Account_TxnCount",
                "mean": "Account_AvgAmount",
                "std": "Account_StdAmount",
                "min": "Account_MinAmount",
                "max": "Account_MaxAmount",
            }
        )
        stats["Account_AmountRange"] = stats["Account_MaxAmount"] - stats["Account_MinAmount"]
        stats["Account_StdAmount"] = stats["Account_StdAmount"].fillna(0)
        df = df.merge(stats, left_on="AccountId", right_index=True, how="left")

    return df


def save_class_distribution(train: pd.DataFrame) -> None:
    counts = train["FraudResult"].value_counts().sort_index()

    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#4C78A8", "#F58518"])
    plt.ylabel("Number of transactions")
    plt.title("Class Distribution in Training Data")

    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=200)
    plt.close()


def save_amount_value_distributions(train: pd.DataFrame) -> None:
    # Use log1p(abs(.)) to handle negatives in Amount
    df = train[["FraudResult", "Amount", "Value"]].copy()
    df["LogAmount"] = np.log1p(np.abs(df["Amount"]))
    df["LogValue"] = np.log1p(np.abs(df["Value"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, col, title in [
        (axes[0], "LogAmount", "log(1+|Amount|)"),
        (axes[1], "LogValue", "log(1+|Value|)"),
    ]:
        ax.hist(df.loc[df["FraudResult"] == 0, col], bins=60, alpha=0.65, label="Legitimate", color="#4C78A8")
        ax.hist(df.loc[df["FraudResult"] == 1, col], bins=60, alpha=0.75, label="Fraud", color="#F58518")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.set_xlabel("Transformed value")
        ax.legend()

    fig.suptitle("Distribution Shift Between Legitimate and Fraud Transactions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "amount_value_distributions.png", dpi=200)
    plt.close(fig)


def save_model_curves_and_threshold_plot(train: pd.DataFrame) -> dict:
    # Feature engineering aligned with the notebook (core engineered + account aggregates)
    train_fe = engineer_features(train, with_account_aggregates=True)

    # IMPORTANT: The notebook’s best LightGBM model was trained on a fixed 17-feature set
    # (confirmed by the stored feature-importance plot). We replicate that exact feature list
    # here to keep the paper figures consistent with the notebook results.
    expected_feature_cols = [
        "Amount",
        "Value",
        "Account_TxnCount",
        "Account_StdAmount",
        "LogAmount",
        "Account_AvgAmount",
        "Account_MinAmount",
        "IsBusinessHour",
        "Account_MaxAmount",
        "Account_AmountRange",
        "PricingStrategy",
        "Amount_Value_Ratio",
        "IsWeekend",
        "IsLateNight",
        "Amount_Value_Difference",
        "Amount_Value_Interaction",
        "LogValue",
    ]
    feature_cols = [c for c in expected_feature_cols if c in train_fe.columns]

    X = train_fe[feature_cols].fillna(0)
    y = train_fe["FraudResult"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train best baseline model from notebook (class_weighted LightGBM)
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_val)[:, 1]

    # Curves
    fpr, tpr, _ = roc_curve(y_val, y_score)
    roc_auc = auc(fpr, tpr)

    p, r, thresholds = precision_recall_curve(y_val, y_score)
    pr_auc = auc(r, p)

    # Find F1-optimal threshold
    f1_scores = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    opt_threshold = float(thresholds[best_idx])

    y_pred_default = (y_score >= 0.5).astype(int)
    y_pred_opt = (y_score >= opt_threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "opt_threshold": opt_threshold,
        "f1_default": float(f1_score(y_val, y_pred_default)),
        "f1_opt": float(f1_score(y_val, y_pred_opt)),
        "precision_opt": float(precision_score(y_val, y_pred_opt)),
        "recall_opt": float(recall_score(y_val, y_pred_opt)),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_features": int(X.shape[1]),
    }

    # Save ROC + PR figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(fpr, tpr, color="#E45756", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve (Validation)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(r, p, color="#4C78A8", linewidth=2, label=f"AUC = {pr_auc:.3f}")
    axes[1].set_title("Precision–Recall Curve (Validation)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    fig.suptitle("LightGBM Performance Curves")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_model_curves.png", dpi=200)
    plt.close(fig)

    # Save threshold trade-off plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(thresholds, p[:-1], label="Precision", color="#4C78A8", linewidth=2)
    axes[0].plot(thresholds, r[:-1], label="Recall", color="#E45756", linewidth=2)
    axes[0].axvline(opt_threshold, color="#54A24B", linestyle="--", linewidth=2)
    axes[0].set_title("Precision/Recall vs Threshold")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].plot(thresholds, f1_scores, color="#54A24B", linewidth=2)
    axes[1].axvline(opt_threshold, color="#54A24B", linestyle="--", linewidth=2)
    axes[1].set_title("F1 vs Threshold")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1")

    fig.suptitle(f"Threshold Optimization (opt = {opt_threshold:.4f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_optimization.png", dpi=200)
    plt.close(fig)

    return metrics


def main() -> None:
    train_path = Path("training_data.csv")
    if not train_path.exists():
        raise FileNotFoundError("training_data.csv not found in the current directory")

    train = pd.read_csv(train_path)

    if "FraudResult" not in train.columns:
        raise ValueError("Expected label column 'FraudResult' not found in training_data.csv")

    save_class_distribution(train)
    save_amount_value_distributions(train)
    metrics = save_model_curves_and_threshold_plot(train)

    # Write a small metrics file for convenience
    out = Path("resources")
    out.mkdir(exist_ok=True)
    (out / "generated_metrics.json").write_text(
        pd.Series(metrics).to_json(indent=2), encoding="utf-8"
    )

    print("Saved figures to figs/ and metrics to resources/generated_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
