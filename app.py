import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm
import plotly.graph_objects as go
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run `run_fraud_detection.py` first.")
        return None, None

model, scaler = load_assets()

# Feature Engineering Function (Simplified for Inference)
def engineer_features(df):
    # 1. Amount and Value interactions
    df['Amount_Value_Ratio'] = df['Amount'] / (df['Value'] + 1e-6)
    df['Amount_Value_Interaction'] = df['Amount'] * df['Value']
    df['Amount_Value_Difference'] = df['Amount'] - df['Value']
    
    # 2. Log transformations
    df['LogAmount'] = np.log1p(np.abs(df['Amount']))
    df['LogValue'] = np.log1p(np.abs(df['Value']))
    
    # 3. Time-based features
    # Ensure columns exist and fill with default if needed (though UI provides them)
    if 'Weekday' not in df.columns:
        df['Weekday'] = 0 # Default
    if 'Hour' not in df.columns:
        df['Hour'] = 12 # Default
        
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    df['IsLateNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    return df

# Main Layout
st.title("🛡️ Mobile Money Fraud Detection System")
st.markdown("### Real-time Transaction Analysis")

# Sidebar
with st.sidebar:
    st.header("System Status")
    if model and scaler: # Check if loaded
        st.success("✅ Model Loaded (LightGBM)")
        st.success("✅ Scaler Loaded")
    else:
        st.error("❌ System Offline")
        
    st.markdown("---")
    st.header("Navigation")
    mode = st.radio("Select Mode", ["Single Prediction", "Batch Analysis"])
    
    st.markdown("---")
    st.info("Developed for Master's Thesis\nDec 2025")

if mode == "Single Prediction":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Transaction Details")
        
        with st.form("prediction_form"):
            # Inputs
            amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0, step=100.0)
            value = st.number_input("Transaction Value", min_value=0.0, value=5000.0, step=100.0)
            
            # Time Inputs
            trans_date = st.date_input("Date", datetime.now())
            trans_time = st.time_input("Time", datetime.now())
            
            # Context Inputs (Used for visual context or if model had categorical feats)
            # Note: My current model mainly uses numericals derived from these or IDs
            # But let's verify feature_cols used in training.
            # The model was trained on: Amount, Value, Hour, Day, Month, Weekday, IsWeekend, etc.
            # It also included "PricingStrategy" if it was numerical?
            # Let's add Pricing Strategy as it was in the top features list in the paper
            pricing_strategy = st.selectbox("Pricing Strategy", [0, 1, 2, 4])
            
            product_category = st.selectbox("Product Category", 
                ["Financial Services", "Airtime", "Utility Bill", "Data Bundles", "Transport", "Other"])
            
            channel_id = st.selectbox("Channel ID", ["Channel_1", "Channel_2", "Channel_3", "Channel_5"])
            
            submitted = st.form_submit_button("Analyze Transaction")
            
    with col2:
        if submitted and model:
            # Prepare Input Data
            dt_combined = datetime.combine(trans_date, trans_time)
            
            input_data = pd.DataFrame({
                'Amount': [amount],
                'Value': [value],
                'PricingStrategy': [pricing_strategy],
                # Time features extracted immediately
                'Hour': [dt_combined.hour],
                'Day': [dt_combined.day],
                'Month': [dt_combined.month],
                'Weekday': [dt_combined.weekday()]
            })
            
            # Engineer Features
            processed_data = engineer_features(input_data)
            
            # Identify columns expected by model
            # To be safe, we need to match the columns the scaler expects.
            # The scaler fits on X_train. 
            # We strictly need to match the feature set. 
            # Based on the script, it selects: 
            # [Amount, Value, PricingStrategy, Hour, Day, Month, Weekday, 
            #  Amount_Value_Ratio, Amount_Value_Interaction, Amount_Value_Difference, 
            #  LogAmount, LogValue, IsWeekend, IsBusinessHour, IsLateNight]
            
            # Let's ensure order doesn't matter for the scaler (it usually processes numpy array, so ORDER MATTERS!)
            # We need to know the exact column order. 
            # Best way is to rely on column names if we convert to dataframe before scaling, 
            # but StandardScaler expects array.
            # I will trust the feature engineering produces them in a similar way, 
            # BUT: 
            # In the script: 
            # feature_cols = [col for col in train_df.columns if ... ]
            # This order depends on pandas column order.
            
            # CRITICAL FIX:
            # I will assume standard order or I need to load the feature names.
            # Since I cannot easily read the feature names from the scaler object directly (unless it's a dataframe scaler),
            # I will attempt to reconstruct the exact same order as the script.
            
            # Scaler features: ['Amount' 'Value' 'PricingStrategy' 'Amount_Value_Ratio'
            # 'Amount_Value_Interaction' 'Amount_Value_Difference' 'LogAmount'
            # 'LogValue' 'IsWeekend' 'IsBusinessHour' 'IsLateNight']
            
            expected_cols = [
                'Amount', 'Value', 'PricingStrategy', 
                'Amount_Value_Ratio', 'Amount_Value_Interaction', 'Amount_Value_Difference',
                'LogAmount', 'LogValue',
                'IsWeekend', 'IsBusinessHour', 'IsLateNight'
            ]
            
            # Filter and reorder
            # Note: The script used `train_feature_cols` logic. 
            # `PricingStrategy` is numerical.
            # Categoricals like ChannelId/ProductCategory were One-Hot Encoded? 
            # Checking script: 
            # "train_feature_cols = [col ... if ... dtype in [np.number...]]"
            # It seems the script DID NOT One-Hot Encode categoricals before selecting features!
            # It only used numerical columns. 
            # So `ProductCategory` and `ChannelId` were likely IGNORED by the model 
            # unless they were label encoded or the script had implicit encoding (LightGBM can handle cats, but sklearn scaler cannot).
            # The script has: 
            # "X = train_df[feature_cols].fillna(0)"
            # And `feature_cols` are only number types.
            # So the model ONLY uses numerical features.
            
            try:
                final_input = processed_data[expected_cols]
                
                # Scale
                final_input_scaled = scaler.transform(final_input)
                
                # Predict
                prediction = model.predict(final_input_scaled)[0]
                probability = model.predict_proba(final_input_scaled)[0][1]
                
                st.subheader("Analysis Result")
                
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if probability > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                    }
                ))
                st.plotly_chart(fig)
                
                # Decision
                if probability > 0.5:
                    st.error(f"🚨 **FRAUD DETECTED** (Confidence: {probability:.1%})")
                    st.warning("Recommended Action: **Block Transaction & Verify User**")
                else:
                    st.success(f"✅ **LEGITIMATE TRANSACTION** (Confidence: {1-probability:.1%})")
                
                # Explainability factors (Simple rules for demo)
                with st.expander("Why this result?"):
                    st.write("Key Risk Factors:")
                    if amount > 50000:
                        st.write("- ⚠️ Very High Transaction Amount")
                    if dt_combined.hour < 6 or dt_combined.hour > 23:
                        st.write("- ⚠️ Late Night Transaction Time")
                    if amount != value:
                        st.write(f"- ℹ️ Fees/Charges Applied (Diff: {amount - value})")
                    
            except KeyError as e:
                st.error(f"Feature Mismatch Error: Missing {e}. The model expects specific columns.")
                st.write("Expected:", expected_cols)
                st.write("Got:", processed_data.columns.tolist())

elif mode == "Batch Analysis":
    st.header("Batch Transaction Processing")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file and model:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} transactions.")
            
            if st.button("Run Batch Analysis"):
                # Preprocess
                # (Need validation if columns exist)
                # For demo, just show placeholder
                st.info("Batch processing logic would go here, replicating the single prediction pipeline loop.")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

