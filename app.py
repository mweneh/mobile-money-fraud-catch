import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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
        # Load stats with AccountId as index
        stats = pd.read_csv('account_stats_artifact.csv', index_col=0)
        with open('feature_cols.json', 'r') as f:
            features = json.load(f)
        with open('fill_values.json', 'r') as f:
            fill_vals = json.load(f)
        return model, stats, features, fill_vals
    except FileNotFoundError as e:
        return None, None, None, None

model, account_stats, feature_cols, fill_values = load_assets()

def engineer_features(df, account_stats, fill_values):
    # 1. Feature Engineering (Replicating Notebook)
    
    # Time-based features
    # Ensure TransactionStartTime is datetime
    if not np.issubdtype(df['TransactionStartTime'].dtype, np.datetime64):
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['Day'] = df['TransactionStartTime'].dt.day
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Weekday'] = df['TransactionStartTime'].dt.weekday
    
    # Interactions
    df['Amount_Value_Ratio'] = df['Amount'] / (df['Value'] + 1e-6)
    df['Amount_Value_Interaction'] = df['Amount'] * df['Value']
    df['Amount_Value_Difference'] = df['Amount'] - df['Value']
    
    # Log Transformations
    df['LogAmount'] = np.log1p(np.abs(df['Amount']))
    df['LogValue'] = np.log1p(np.abs(df['Value']))
    
    # Time flags
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    df['IsLateNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    # 2. Merge Account Stats
    if 'AccountId' in df.columns and account_stats is not None:
        df = df.merge(account_stats, left_on='AccountId', right_index=True, how='left')
        
    # 3. Fill Missing Values (for new users)
    if fill_values:
        for col, val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
            
    return df

# Main Layout
st.title("🛡️ Mobile Money Fraud Detection System")
st.markdown("### Real-time Transaction Analysis")

# Sidebar
with st.sidebar:
    st.header("System Status")
    if model:
        st.success("✅ Model Loaded")
        st.success(f"✅ Historical Stats: {len(account_stats):,} Accounts")
    else:
        st.error("❌ System Offline (Artifacts Missing)")
        st.info("Please run the notebook to generate 'model.pkl', 'account_stats_artifact.csv', and/or 'feature_cols.json'.")
        
    st.markdown("---")
    st.header("Navigation")
    mode = st.radio("Select Mode", ["Single Prediction", "Batch Analysis"])
    

if mode == "Single Prediction":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Transaction Details")
        with st.form("prediction_form"):
            # Core Inputs
            account_id = st.text_input("Account ID", "AccountId_1234")
            amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
            value = st.number_input("Value", min_value=0.0, value=5000.0, step=100.0)
            pricing_strategy = st.selectbox("Pricing Strategy", [0, 1, 2, 4])
            
            # Time Inputs
            trans_date = st.date_input("Date", datetime.now())
            trans_time = st.time_input("Time", datetime.now())
            
            submitted = st.form_submit_button("Analyze Transaction")
            
    with col2:
        if submitted and model:
            # Create Input DataFrame
            dt_combined = datetime.combine(trans_date, trans_time)
            
            input_data = pd.DataFrame({
                'AccountId': [account_id],
                'Amount': [amount],
                'Value': [value],
                'PricingStrategy': [pricing_strategy],
                'TransactionStartTime': [dt_combined], # Will be parsed in engineer_features
                # Add dummy columns for other ID fields if needed by simple validation
            })
            
            # Process
            try:
                processed_data = engineer_features(input_data, account_stats, fill_values)
                
                # Check for feature columns
                if feature_cols:
                    # Ensure all feature cols exist
                    for col in feature_cols:
                        if col not in processed_data.columns:
                            processed_data[col] = 0.0 # Default missing features to 0
                    
                    # Select exact columns in order
                    final_X = processed_data[feature_cols]
                else:
                    final_X = processed_data.select_dtypes(include=[np.number])
                    st.warning("Feature columns list not found. Using all numeric columns.")
                
                # Predict
                prob = model.predict_proba(final_X)[0][1]
                
                # Display Result
                st.subheader("Analysis Result")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': "Fraud Probability (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if prob > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "#ffcccb"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90 # High risk threshold example
                        }
                    }
                ))
                st.plotly_chart(fig)
                
                if prob > 0.5:
                    st.error(f"🚨 FRAUD DETECTED")
                    st.write(f"Confidence: **{prob:.2%}**")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
                    st.write(f"Confidence: **{1-prob:.2%}**")
                    
                with st.expander("Debug Info"):
                    st.write("Processed Features:", final_X)
                    
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
                import traceback
                st.write(traceback.format_exc())

elif mode == "Batch Analysis":
    st.info("Batch analysis feature coming soon.")
