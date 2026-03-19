"""
Monitoring Dashboard
Real-time Streamlit dashboard for ML system monitoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from src.monitoring.metrics_collector import MetricsCollector

# Page configuration
st.set_page_config(
    page_title="ML Commit Risk Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_prediction_data(limit=1000):
    """Load prediction data from logs"""
    collector = MetricsCollector()
    df = collector.load_predictions(limit=limit)
    return df

def create_risk_distribution_chart(df):
    """Create risk level distribution pie chart"""
    if df.empty:
        return None
    
    risk_counts = df['risk_level'].value_counts()
    
    colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#44ff44'}
    color_sequence = [colors.get(level, '#999999') for level in risk_counts.index]
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Risk Level Distribution',
        color_discrete_sequence=color_sequence
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_risk_score_histogram(df):
    """Create risk score distribution histogram"""
    if df.empty:
        return None
    
    fig = px.histogram(
        df,
        x='risk_score',
        nbins=20,
        title='Risk Score Distribution',
        labels={'risk_score': 'Risk Score', 'count': 'Number of Commits'}
    )
    
    # Add threshold lines
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Risk", annotation_position="top")
    fig.add_vline(x=0.4, line_dash="dash", line_color="orange",
                  annotation_text="Medium Risk", annotation_position="top")
    
    return fig

def create_predictions_timeline(df):
    """Create predictions over time chart"""
    if df.empty:
        return None
    
    # Group by hour
    df['hour'] = df['timestamp'].dt.floor('H')
    hourly = df.groupby(['hour', 'risk_level']).size().reset_index(name='count')
    
    fig = px.line(
        hourly,
        x='hour',
        y='count',
        color='risk_level',
        title='Predictions Over Time',
        labels={'hour': 'Time', 'count': 'Number of Predictions'},
        color_discrete_map={'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#44ff44'}
    )
    
    return fig

def create_response_time_chart(df):
    """Create response time chart"""
    if df.empty or 'response_time_ms' not in df.columns:
        return None
    
    fig = px.box(
        df,
        y='response_time_ms',
        title='API Response Time Distribution',
        labels={'response_time_ms': 'Response Time (ms)'}
    )
    
    return fig


# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">🔍 ML Commit Risk Monitoring Dashboard</div>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        
        data_limit = st.selectbox(
            "Data points to display",
            [100, 500, 1000, 5000],
            index=2
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info(
            "This dashboard monitors the ML commit risk scoring system. "
            "It displays real-time predictions, model performance metrics, "
            "and system health indicators."
        )
        
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_prediction_data(limit=data_limit)
    
    # Check if data exists
    if df.empty:
        st.warning("⚠️ No prediction data available yet. Make some predictions via the API first!")
        st.code("""
# Example: Make a prediction
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "commit_hash": "test123",
    "lines_added": 150,
    "lines_deleted": 50,
    "files_changed": 8,
    "touches_core": 1,
    "total_commits": 100,
    "buggy_commits": 25
  }'
        """)
        return
    
    # Summary metrics
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Predictions",
            value=len(df),
            delta=f"+{len(df[df['timestamp'] > datetime.now() - timedelta(hours=1)])} last hour"
        )
    
    with col2:
        avg_risk = df['risk_score'].mean()
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.2%}",
            delta=f"{avg_risk - 0.5:.2%} from baseline"
        )
    
    with col3:
        high_risk_count = (df['risk_level'] == 'HIGH').sum()
        high_risk_pct = high_risk_count / len(df) * 100
        st.metric(
            label="High Risk Commits",
            value=high_risk_count,
            delta=f"{high_risk_pct:.1f}% of total"
        )
    
    with col4:
        if 'response_time_ms' in df.columns:
            avg_response = df['response_time_ms'].mean()
            st.metric(
                label="Avg Response Time",
                value=f"{avg_response:.1f}ms",
                delta="✅ Good" if avg_response < 100 else "⚠️ Slow"
            )
        else:
            st.metric(label="Avg Response Time", value="N/A")
    
    st.markdown("---")
    
    # Charts
    st.header("📊 Visualizations")
    
    # Row 1: Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_dist_chart = create_risk_distribution_chart(df)
        if risk_dist_chart:
            st.plotly_chart(risk_dist_chart, use_container_width=True)
    
    with col2:
        risk_hist_chart = create_risk_score_histogram(df)
        if risk_hist_chart:
            st.plotly_chart(risk_hist_chart, use_container_width=True)
    
    # Row 2: Timeline and response time
    col1, col2 = st.columns(2)
    
    with col1:
        timeline_chart = create_predictions_timeline(df)
        if timeline_chart:
            st.plotly_chart(timeline_chart, use_container_width=True)
    
    with col2:
        response_chart = create_response_time_chart(df)
        if response_chart:
            st.plotly_chart(response_chart, use_container_width=True)
        else:
            st.info("Response time data not available")
    
    st.markdown("---")
    
    # Recent predictions table
    st.header("🔍 Recent Predictions")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['HIGH', 'MEDIUM', 'LOW'],
            default=['HIGH', 'MEDIUM', 'LOW']
        )
    
    with col2:
        show_count = st.slider("Show last N predictions", 5, 50, 10)
    
    # Apply filters
    filtered_df = df[df['risk_level'].isin(risk_filter)].tail(show_count)
    
    # Display table
    display_cols = ['timestamp', 'commit_hash', 'risk_score', 'risk_level']
    if 'response_time_ms' in filtered_df.columns:
        display_cols.append('response_time_ms')
    
    st.dataframe(
        filtered_df[display_cols].sort_values('timestamp', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ==============================================================================
# RUN DASHBOARD
# ==============================================================================

if __name__ == "__main__":
    main()