import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🔍 ML Commit Risk Monitoring")

# Check if log file exists
log_file = Path("logs/predictions.log")

if log_file.exists():
    st.success(f"✅ Log file found: {log_file}")
    
    # Try to read it
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    st.write(f"Total log entries: {len(lines)}")
    
    # Show raw content
    st.text_area("Raw log content", "\n".join(lines[:5]), height=200)
else:
    st.error("❌ Log file not found!")
    st.write(f"Looking for: {log_file.absolute()}")