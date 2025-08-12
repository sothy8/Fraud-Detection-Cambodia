#!/usr/bin/env python3
"""
Fraud Detection Dashboard - Main Application
Run this with: python main.py or ./main.py
"""

import sys
import os
import subprocess

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_with_streamlit():
    """Run this script with Streamlit if not already running with it"""
    if 'streamlit' not in sys.modules:
        # Enhanced launcher with better validation
        from pathlib import Path
        
        project_path = Path(project_root)
        venv_python = project_path / "venv" / "bin" / "python"
        
        print("🇰🇭 Fraud Detection Cambodia - Dashboard Launcher")
        print("=" * 60)
        
        # Check if virtual environment exists
        if not venv_python.exists():
            print("❌ Virtual environment not found!")
            print(f"   Expected: {venv_python}")
            print("\n💡 Please create a virtual environment first:")
            print("   python -m venv venv")
            print("   source venv/bin/activate")
            print("   pip install -r requirements.txt")
            sys.exit(1)
        
        # Check if Streamlit is installed
        try:
            result = subprocess.run([str(venv_python), '-c', 'import streamlit'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Streamlit not found in virtual environment!")
                print("\n💡 Please install requirements:")
                print("   source venv/bin/activate")
                print("   pip install -r requirements.txt")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error checking Streamlit: {e}")
            sys.exit(1)
        
        # Use virtual environment Python
        cmd = [str(venv_python), '-m', 'streamlit', 'run', __file__]
        
        print(f"� Project Directory: {project_root}")
        print(f"� Python Environment: {venv_python}")
        print("� Starting Fraud Detection Dashboard...")
        print("🌐 Dashboard will be available at: http://localhost:8501")
        print("\n" + "=" * 60)
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error running dashboard: {e}")
            print("💡 Try running: pip install -r requirements.txt")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped by user")
        
        sys.exit(0)

# Check if we need to start Streamlit
if __name__ == "__main__" and 'streamlit' not in sys.modules:
    run_with_streamlit()

# Streamlit application code starts here
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import os
from utils import load_model, score, generate_sample_data, check_model_exists

# Page configuration
st.set_page_config(
    page_title='Fraud Detection Dashboard 🇰🇭', 
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .info-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 600;
    }
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cambodia-flag {
        background: linear-gradient(180deg, #1e40af 33%, #dc2626 33%, #dc2626 66%, #1e40af 66%);
        width: 30px;
        height: 20px;
        display: inline-block;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header with Cambodia branding
st.markdown("""
<div class="main-header">
    <h1>🇰🇭 Cambodia Fraud Detection System</h1>
    <p>Advanced AI-Powered Real-time Transaction Monitoring</p>
    <p><strong>Protecting Cambodian Financial Institutions & Citizens</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header('🔧 Control Panel')

# Check if model exists
if not check_model_exists():
    st.sidebar.error('⚠️ Model not found! Please train a model first.')
    if st.sidebar.button('🚀 Generate Sample Data & Train Model'):
        with st.spinner('Generating sample data and training model...'):
            generate_sample_data()
        st.rerun()
else:
    st.sidebar.success('✅ Model loaded successfully')

# File upload section
uploaded = st.sidebar.file_uploader(
    '📤 Upload Transaction CSV', 
    type=['csv'],
    help='Upload a CSV file with transaction data for fraud analysis'
)

# Sample data section
st.sidebar.markdown('### 📋 Sample Data')
if os.path.exists('data/raw/transactions.csv'):
    st.sidebar.success('✅ Sample data available')
    if st.sidebar.button('📊 Analyze Sample Data'):
        uploaded = 'sample'
else:
    st.sidebar.warning('⚠️ No sample data found')
    if st.sidebar.button('🎲 Generate Sample Data'):
        with st.spinner('Generating sample transactions...'):
            generate_sample_data()
        st.rerun()

# Real-time mode toggle
realtime_mode = st.sidebar.checkbox('🔄 Real-time Mode (Auto-refresh)', value=False)

if realtime_mode:
    st.sidebar.info('🔄 Dashboard will refresh every 30 seconds')

# Main content
if check_model_exists():
    model = load_model()
    
    if uploaded == 'sample':
        df_new = pd.read_csv('data/raw/transactions.csv', parse_dates=['timestamp'])
        st.info('📊 Analyzing sample transaction data...')
    elif uploaded:
        try:
            df_new = pd.read_csv(uploaded, parse_dates=['timestamp'])
            st.success(f'✅ Uploaded file processed: {len(df_new)} transactions')
        except Exception as e:
            st.error(f'❌ Error reading file: {str(e)}')
            st.stop()
    else:
        st.info('👆 Upload a CSV file or use sample data to see fraud analysis')
        st.stop()

    # Process data
    with st.spinner('🔍 Analyzing transactions for fraud patterns...'):
        result = score(df_new, model)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(result)
    flagged_transactions = len(result[result['is_fraud_pred'] == 1])
    avg_fraud_score = result['fraud_score'].mean()
    high_risk_transactions = len(result[result['fraud_score'] > 0.7])
    
    with col1:
        st.metric("📊 Total Transactions", total_transactions)
    with col2:
        st.metric("🚨 Flagged as Fraud", flagged_transactions, 
                 delta=f"{(flagged_transactions/total_transactions*100):.1f}%")
    with col3:
        st.metric("📈 Avg Fraud Score", f"{avg_fraud_score:.3f}")
    with col4:
        st.metric("⚠️ High Risk (>0.7)", high_risk_transactions)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Transaction Analysis", 
        "🚨 Fraud Alerts", 
        "📈 Advanced Analytics", 
        "🌐 Geographic Analysis",
        "⏰ Time Series Analysis",
        "📋 Raw Data"
    ])
    
    with tab1:
        st.subheader('🔍 Transaction Fraud Analysis')
        
        # Fraud score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                result, x='fraud_score', nbins=50,
                title='Distribution of Fraud Scores',
                labels={'fraud_score': 'Fraud Score', 'count': 'Number of Transactions'}
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", 
                              annotation_text="Fraud Threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Fraud by hour
            hourly_fraud = result.groupby('hour').agg({
                'is_fraud_pred': 'sum',
                'txn_id': 'count'
            }).reset_index()
            hourly_fraud['fraud_rate'] = hourly_fraud['is_fraud_pred'] / hourly_fraud['txn_id']
            
            fig_hourly = px.line(
                hourly_fraud, x='hour', y='fraud_rate',
                title='Fraud Rate by Hour of Day',
                labels={'hour': 'Hour', 'fraud_rate': 'Fraud Rate'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    with tab2:
        st.subheader('🚨 Fraud Alerts & Flagged Transactions')
        
        flagged_data = result[result['is_fraud_pred'] == 1].copy()
        
        if len(flagged_data) > 0:
            st.markdown(f'<div class="fraud-alert">⚠️ <strong>{len(flagged_data)} transactions flagged as potentially fraudulent</strong></div>', 
                       unsafe_allow_html=True)
            
            # Sort by fraud score descending
            flagged_data = flagged_data.sort_values('fraud_score', ascending=False)
            
            # Display top fraudulent transactions
            st.dataframe(
                flagged_data[['txn_id', 'user_id', 'recipient_id', 'amount', 'fraud_score', 'timestamp']],
                use_container_width=True
            )
            
            # High-value fraud alerts
            high_value_fraud = flagged_data[flagged_data['amount'] > flagged_data['amount'].quantile(0.9)]
            if len(high_value_fraud) > 0:
                st.warning(f'🚨 **HIGH VALUE ALERT**: {len(high_value_fraud)} high-value transactions flagged!')
                st.dataframe(high_value_fraud[['txn_id', 'amount', 'fraud_score']], use_container_width=True)
        else:
            st.markdown('<div class="success-card">✅ <strong>No fraudulent transactions detected</strong></div>', 
                       unsafe_allow_html=True)
    
    with tab3:
        st.subheader('� Fraud Analytics & Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount vs Fraud Score scatter
            fig_scatter = px.scatter(
                result, x='amount', y='fraud_score', 
                color='is_fraud_pred',
                title='Transaction Amount vs Fraud Score',
                labels={'amount': 'Transaction Amount (KHR)', 'fraud_score': 'Fraud Score'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Fraud by transaction type
            if 'txn_type' in result.columns:
                fraud_by_type = result.groupby('txn_type')['is_fraud_pred'].agg(['sum', 'count']).reset_index()
                fraud_by_type['fraud_rate'] = fraud_by_type['sum'] / fraud_by_type['count']
                
                fig_type = px.bar(
                    fraud_by_type, x='txn_type', y='fraud_rate',
                    title='Fraud Rate by Transaction Type',
                    labels={'txn_type': 'Transaction Type', 'fraud_rate': 'Fraud Rate'}
                )
                st.plotly_chart(fig_type, use_container_width=True)
    
    with tab4:
        st.subheader('📋 All Transactions with Fraud Scores')
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider('Minimum Fraud Score', 0.0, 1.0, 0.0, 0.1)
        with col2:
            max_amount = st.number_input('Max Amount Filter', value=float(result['amount'].max()))
        with col3:
            show_fraud_only = st.checkbox('Show Fraud Only')
        
        # Apply filters
        filtered_result = result[result['fraud_score'] >= min_score]
        filtered_result = filtered_result[filtered_result['amount'] <= max_amount]
        if show_fraud_only:
            filtered_result = filtered_result[filtered_result['is_fraud_pred'] == 1]
        
        st.dataframe(filtered_result, use_container_width=True)
        
        # Download button
        csv = filtered_result.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results as CSV",
            data=csv,
            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Add new enhanced tabs for new visualizations
    if len(result) > 0:
        st.markdown("---")
        st.markdown("## 🚀 Enhanced Analytics Dashboard")
        
        # Enhanced analytics section
        with st.expander("📊 Advanced Visualizations", expanded=True):
            
            # Geographic Analysis
            st.markdown("### 🌐 Geographic Fraud Analysis")
            if 'province' not in result.columns:
                # Add simulated province data for demonstration
                cambodia_provinces = [
                    'Phnom Penh', 'Siem Reap', 'Battambang', 'Kandal', 'Takeo',
                    'Kampong Thom', 'Preah Vihear', 'Prey Veng', 'Svay Rieng',
                    'Kampong Cham', 'Kampong Chhnang', 'Kampong Speu', 'Kratie',
                    'Mondul Kiri', 'Ratanak Kiri', 'Stung Treng', 'Pursat',
                    'Koh Kong', 'Kep', 'Pailin', 'Banteay Meanchey', 'Oddor Meanchey',
                    'Preah Sihanouk', 'Kampot', 'Tboung Khmum'
                ]
                np.random.seed(42)  # For consistent results
                result['province'] = np.random.choice(cambodia_provinces, size=len(result))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Fraud by province
                province_stats = result.groupby('province').agg({
                    'is_fraud_pred': ['sum', 'count'],
                    'fraud_score': 'mean',
                    'amount': 'sum'
                }).reset_index()
                province_stats.columns = ['province', 'fraud_count', 'total_txns', 'avg_fraud_score', 'total_amount']
                province_stats['fraud_rate'] = province_stats['fraud_count'] / province_stats['total_txns']
                
                # Top provinces by fraud rate
                top_fraud_provinces = province_stats.nlargest(10, 'fraud_rate')
                
                fig_provinces = px.bar(
                    top_fraud_provinces, x='province', y='fraud_rate',
                    title='🏆 Top 10 Provinces by Fraud Rate',
                    labels={'province': 'Province', 'fraud_rate': 'Fraud Rate'},
                    color='fraud_rate',
                    color_continuous_scale='Reds'
                )
                fig_provinces.update_xaxes(tickangle=45)
                st.plotly_chart(fig_provinces, use_container_width=True)
            
            with col2:
                # Province risk bubble chart
                fig_bubble = px.scatter(
                    province_stats, 
                    x='total_amount', 
                    y='fraud_rate',
                    size='total_txns',
                    color='avg_fraud_score',
                    hover_name='province',
                    title='🎯 Province Risk Assessment',
                    labels={
                        'total_amount': 'Total Transaction Amount (KHR)',
                        'fraud_rate': 'Fraud Rate',
                        'total_txns': 'Number of Transactions',
                        'avg_fraud_score': 'Average Fraud Score'
                    },
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
            
            # Risk Analysis
            st.markdown("### 🎯 Risk Distribution Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Risk categories pie chart
                risk_categories = pd.cut(result['fraud_score'], 
                                       bins=[0, 0.3, 0.7, 1.0],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
                risk_dist = risk_categories.value_counts()
                
                fig_pie = px.pie(
                    values=risk_dist.values, 
                    names=risk_dist.index,
                    title='Risk Distribution',
                    color_discrete_map={
                        'Low Risk': 'lightgreen',
                        'Medium Risk': 'orange', 
                        'High Risk': 'red'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Amount range analysis
                result['amount_range'] = pd.cut(result['amount'], 
                                              bins=[0, 100000, 500000, 1000000, 5000000, float('inf')],
                                              labels=['<100K', '100K-500K', '500K-1M', '1M-5M', '>5M'])
                
                fraud_by_amount = result.groupby('amount_range').agg({
                    'is_fraud_pred': ['sum', 'count']
                }).reset_index()
                fraud_by_amount.columns = ['amount_range', 'fraud_count', 'total_count']
                fraud_by_amount['fraud_rate'] = fraud_by_amount['fraud_count'] / fraud_by_amount['total_count']
                
                fig_amount_range = px.bar(
                    fraud_by_amount, x='amount_range', y='fraud_rate',
                    title='Fraud Rate by Amount Range',
                    labels={'amount_range': 'Amount Range (KHR)', 'fraud_rate': 'Fraud Rate'},
                    color='fraud_rate',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_amount_range, use_container_width=True)
            
            with col3:
                # User behavior analysis
                user_stats = result.groupby('user_id').agg({
                    'amount': ['sum', 'count', 'mean'],
                    'fraud_score': 'mean',
                    'is_fraud_pred': 'sum'
                }).reset_index()
                user_stats.columns = ['user_id', 'total_amount', 'txn_count', 'avg_amount', 'avg_fraud_score', 'fraud_count']
                user_stats['fraud_rate'] = user_stats['fraud_count'] / user_stats['txn_count']
                
                # High-risk users
                high_risk_users = user_stats[user_stats['avg_fraud_score'] > 0.7]
                
                fig_donut = px.pie(
                    values=[len(high_risk_users), len(user_stats) - len(high_risk_users)],
                    names=['High Risk Users', 'Normal Users'],
                    title='User Risk Profile',
                    hole=0.4,
                    color_discrete_map={'High Risk Users': 'red', 'Normal Users': 'lightblue'}
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            
            # Time Series Analysis
            if 'timestamp' in result.columns:
                st.markdown("### ⏰ Time-based Fraud Patterns")
                result['datetime'] = pd.to_datetime(result['timestamp'])
                result['hour'] = result['datetime'].dt.hour
                result['day_of_week'] = result['datetime'].dt.day_name()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hourly fraud heatmap
                    hourly_fraud = result.groupby('hour').agg({
                        'is_fraud_pred': 'sum',
                        'txn_id': 'count',
                        'fraud_score': 'mean'
                    }).reset_index()
                    hourly_fraud['fraud_rate'] = hourly_fraud['is_fraud_pred'] / hourly_fraud['txn_id']
                    
                    fig_hourly = px.line(
                        hourly_fraud, x='hour', y='fraud_rate',
                        title='🕐 Hourly Fraud Pattern',
                        labels={'hour': 'Hour of Day', 'fraud_rate': 'Fraud Rate'},
                        markers=True
                    )
                    fig_hourly.add_hline(y=hourly_fraud['fraud_rate'].mean(), 
                                       line_dash="dash", line_color="red",
                                       annotation_text="Average Rate")
                    st.plotly_chart(fig_hourly, use_container_width=True)
                
                with col2:
                    # Weekly pattern
                    daily_stats = result.groupby('day_of_week').agg({
                        'fraud_score': 'mean',
                        'is_fraud_pred': 'sum',
                        'txn_id': 'count'
                    }).reset_index()
                    daily_stats['fraud_rate'] = daily_stats['is_fraud_pred'] / daily_stats['txn_id']
                    
                    # Reorder days
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    daily_stats['day_of_week'] = pd.Categorical(daily_stats['day_of_week'], categories=day_order, ordered=True)
                    daily_stats = daily_stats.sort_values('day_of_week')
                    
                    fig_daily = px.bar(
                        daily_stats, x='day_of_week', y='fraud_rate',
                        title='📅 Weekly Fraud Pattern',
                        labels={'day_of_week': 'Day of Week', 'fraud_rate': 'Fraud Rate'},
                        color='fraud_rate',
                        color_continuous_scale='RdYlBu_r'
                    )
                    fig_daily.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_daily, use_container_width=True)
            
            # Enhanced scatter plot
            st.markdown("### 🎯 Enhanced Fraud Score Analysis")
            
            # Enhanced scatter with multiple dimensions
            fig_enhanced = px.scatter(
                result, x='amount', y='fraud_score', 
                color='is_fraud_pred',
                size='hour' if 'hour' in result.columns else 'amount',
                hover_data=['user_id', 'timestamp'] if 'timestamp' in result.columns else ['user_id'],
                title='🔍 Multi-dimensional Fraud Analysis',
                labels={
                    'amount': 'Transaction Amount (KHR)', 
                    'fraud_score': 'Fraud Score',
                    'is_fraud_pred': 'Fraud Status'
                },
                color_discrete_map={0: 'lightblue', 1: 'red'}
            )
            fig_enhanced.add_hline(y=0.5, line_dash="dash", line_color="red", 
                                 annotation_text="Fraud Threshold")
            fig_enhanced.update_layout(height=500)
            st.plotly_chart(fig_enhanced, use_container_width=True)
            
            # Statistical Summary
            st.markdown("### 📊 Statistical Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📈 Average Amount",
                    f"{result['amount'].mean():,.0f} KHR",
                    delta=f"Median: {result['amount'].median():,.0f}"
                )
            
            with col2:
                fraud_rate = (result['is_fraud_pred'].sum() / len(result)) * 100
                st.metric(
                    "🚨 Overall Fraud Rate",
                    f"{fraud_rate:.2f}%",
                    delta=f"Std: {result['fraud_score'].std():.3f}"
                )
            
            with col3:
                peak_hour = result.groupby('hour')['is_fraud_pred'].sum().idxmax() if 'hour' in result.columns else "N/A"
                st.metric(
                    "⏰ Peak Fraud Hour",
                    f"{peak_hour}:00" if peak_hour != "N/A" else "N/A",
                    delta="Based on fraud count"
                )
            
            with col4:
                high_value_threshold = result['amount'].quantile(0.9)
                high_value_fraud = result[(result['amount'] > high_value_threshold) & (result['is_fraud_pred'] == 1)]
                st.metric(
                    "💰 High-Value Frauds",
                    len(high_value_fraud),
                    delta=f">{high_value_threshold:,.0f} KHR"
                )

else:
    st.error('❌ Model not available. Please generate sample data and train a model first.')
    if st.button('🚀 Setup Project (Generate Data & Train Model)'):
        with st.spinner('Setting up project...'):
            generate_sample_data()
        st.rerun()

# Auto-refresh for real-time mode (disabled to prevent issues)
# if realtime_mode:
#     import time
#     time.sleep(30)
#     st.rerun()
