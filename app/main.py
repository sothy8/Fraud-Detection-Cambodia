import streamlit as st
import pandas as pd
from utils import load_model, score

st.set_page_config(page_title='Fraud Detection Dashboard 🇰🇭', layout='wide')
st.title('📊 Real‑Time Fraud Detection for Mobile Wallets')

uploaded = st.sidebar.file_uploader('Upload new transactions CSV', type=['csv'])
sample_link = 'data/raw/transactions.csv'
st.sidebar.markdown(f'[Download sample transactions]({sample_link})')

model = load_model()

if uploaded:
    df_new = pd.read_csv(uploaded, parse_dates=['timestamp'])
    result = score(df_new, model)
    st.subheader('All Transactions with Fraud Scores')
    st.dataframe(result[['txn_id','user_id','recipient_id','amount','fraud_score','is_fraud_pred']])

    st.subheader('🚨 Flagged Transactions')
    st.dataframe(result[result['is_fraud_pred']==1][['txn_id','user_id','amount','fraud_score']])
else:
    st.info('Upload a CSV to see fraud scores. Or run simulation & training first.')
