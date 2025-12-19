# Simulated real‑time transaction scoring demo
# This script re‑uses the exact feature‑engineering, scaler, and model
# from the Streamlit app, but runs a short loop (5 iterations) to show
# how a single transaction can be processed instantly.

import time
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------------------------------------------
# 1️⃣ Load model & scaler (same as app.py)
# ----------------------------------------------------------------------
model_path = Path("model.pkl")
scaler_path = Path("scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ----------------------------------------------------------------------
# 2️⃣ Feature‑engineering (copy from app.py)
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Amount_Value_Ratio"] = df["Amount"] / (df["Value"] + 1e-6)
    df["Amount_Value_Interaction"] = df["Amount"] * df["Value"]
    df["Amount_Value_Difference"] = df["Amount"] - df["Value"]
    df["LogAmount"] = np.log1p(df["Amount"])
    df["LogValue"] = np.log1p(df["Value"])
    # Temporal flags – we need the hour column for these
    df["IsWeekend"] = df["Hour"].isin([0, 6]).astype(int)
    df["IsBusinessHour"] = df["Hour"].between(9, 17).astype(int)
    df["IsLateNight"] = df["Hour"].between(0, 5).astype(int)
    return df

# ----------------------------------------------------------------------
# 3️⃣ Expected column order (must match scaler training)
# ----------------------------------------------------------------------
EXPECTED_COLS = [
    "Amount", "Value", "PricingStrategy",
    "Amount_Value_Ratio", "Amount_Value_Interaction", "Amount_Value_Difference",
    "LogAmount", "LogValue",
    "IsWeekend", "IsBusinessHour", "IsLateNight",
]

# ----------------------------------------------------------------------
# 4️⃣ Helper to generate a random transaction (replace with real source if desired)
# ----------------------------------------------------------------------
def random_transaction():
    amount = round(random.uniform(10, 5000), 2)
    value = round(amount * random.uniform(0.9, 1.1), 2)
    pricing = random.choice([0, 1, 2, 3, 4])
    hour = random.randint(0, 23)
    return pd.DataFrame({
        "Amount": [amount],
        "Value": [value],
        "PricingStrategy": [pricing],
        "Hour": [hour],
    })

# ----------------------------------------------------------------------
# 5️⃣ Real‑time loop (5 iterations for demo)
# ----------------------------------------------------------------------
print("🔁 Starting simulated real‑time scoring (5 transactions)…")
for i in range(5):
    raw = random_transaction()
    engineered = engineer_features(raw)
    X = engineered[EXPECTED_COLS]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1][0]
    label = "FRAUD" if prob >= 0.5 else "LEGIT"
    print(
        f"[{time.strftime('%H:%M:%S')}] Transaction {i+1}: "
        f"Amount={raw['Amount'][0]:.2f}, Hour={raw['Hour'][0]}, "
        f"Prob={prob:.3f} → {label}"
    )
    time.sleep(1)  # pause to mimic arrival spacing
print("✅ Demo finished.")
