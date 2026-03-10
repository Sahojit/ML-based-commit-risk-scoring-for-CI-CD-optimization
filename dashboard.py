import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="ML Commit Risk Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00cc66; font-weight: bold; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_predictions(log_path: str) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "features" in df.columns:
        features_df = pd.json_normalize(df["features"])
        features_df.index = df.index
        overlap = features_df.columns.intersection(df.columns)
        features_df = features_df.drop(columns=overlap)
        df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)

    return df

@st.cache_resource
def get_predictor():
    try:
        from src.inference.predictor import CommitPredictor
        return CommitPredictor()
    except Exception as e:
        return None

st.sidebar.title("🔍 ML Commit Risk")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Live Prediction", "Model Info"],
    index=0,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

log_file = "logs/predictions.log"
df_all = load_predictions(log_file)

if not df_all.empty:
    min_date = pd.Timestamp(df_all["timestamp"].min()).date()
    max_date = pd.Timestamp(df_all["timestamp"].max()).date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    risk_options = ["ALL"] + sorted(df_all["risk_level"].unique().tolist())
    selected_risk = st.sidebar.multiselect(
        "Risk level", options=risk_options, default=["ALL"]
    )

    search_hash = st.sidebar.text_input("Search commit hash", "")

    df = df_all.copy()
    if len(date_range) == 2:
        start, end = date_range
        df = df[
            (df["timestamp"].dt.date >= start)
            & (df["timestamp"].dt.date <= end)
        ]
    if "ALL" not in selected_risk and selected_risk:
        df = df[df["risk_level"].isin(selected_risk)]
    if search_hash:
        df = df[df["commit_hash"].str.contains(search_hash, case=False, na=False)]
else:
    df = df_all

st.sidebar.markdown("---")
auto_refresh = st.sidebar.toggle("Auto-refresh (30s)", value=False)
if auto_refresh:
    st.sidebar.caption("Dashboard refreshes every 30 seconds")
    import time
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if time.time() - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

if page == "Dashboard":
    st.title("ML Commit Risk Monitoring")

    if df.empty:
        st.warning("No prediction data found. Run the API and make some predictions first, or use the **Live Prediction** tab to generate data.")
        st.info("Start the API with: `python scripts/run_api.py`")
        st.stop()

    st.caption(f"Showing {len(df)} of {len(df_all)} predictions")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Total Predictions", len(df))
    with c2:
        avg_risk = df["risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.2%}")
    with c3:
        high = (df["risk_level"] == "HIGH").sum()
        st.metric("High Risk", high, delta=None)
    with c4:
        med = (df["risk_level"] == "MEDIUM").sum()
        st.metric("Medium Risk", med)
    with c5:
        low = (df["risk_level"] == "LOW").sum()
        st.metric("Low Risk", low)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        risk_counts = df["risk_level"].value_counts()
        color_map = {"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#00cc66"}
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=color_map,
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+value", pull=[0.05] * len(risk_counts))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Risk Score Over Time")
        df_sorted = df.sort_values("timestamp")
        fig_line = px.scatter(
            df_sorted,
            x="timestamp",
            y="risk_score",
            color="risk_level",
            color_discrete_map=color_map,
            hover_data=["commit_hash"],
        )
        fig_line.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="HIGH threshold")
        fig_line.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="MEDIUM threshold")
        fig_line.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_line, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(
            df, x="risk_score", nbins=25,
            color_discrete_sequence=["#636EFA"],
            marginal="box",
        )
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red")
        fig_hist.add_vline(x=0.4, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        if "response_time_ms" in df.columns:
            st.subheader("API Response Time (ms)")
            fig_resp = px.line(
                df.sort_values("timestamp"),
                x="timestamp",
                y="response_time_ms",
                markers=True,
            )
            avg_rt = df["response_time_ms"].mean()
            fig_resp.add_hline(y=avg_rt, line_dash="dot", annotation_text=f"avg {avg_rt:.1f}ms")
            st.plotly_chart(fig_resp, use_container_width=True)
        else:
            st.subheader("Risk by Hour of Day")
            if "hour_of_day" in df.columns:
                hourly = df.groupby("hour_of_day")["risk_score"].mean().reset_index()
                fig_hour = px.bar(hourly, x="hour_of_day", y="risk_score",
                                  labels={"hour_of_day": "Hour", "risk_score": "Avg Risk"})
                st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("---")

    numeric_cols = [c for c in ["lines_added", "lines_deleted", "files_changed",
                                "total_churn", "bug_rate", "risk_score"] if c in df.columns]
    if len(numeric_cols) >= 3:
        with st.expander("Feature Correlation Heatmap", expanded=False):
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Prediction Details")

    display_cols = [c for c in ["timestamp", "commit_hash", "risk_score", "risk_level",
                                "lines_added", "lines_deleted", "files_changed",
                                "response_time_ms"] if c in df.columns]

    sort_col = st.selectbox("Sort by", display_cols, index=0)
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

    df_display = df[display_cols].sort_values(
        sort_col, ascending=(sort_order == "Ascending")
    )

    def highlight_risk(row):
        colors = {"HIGH": "background-color: #ff4b4b33",
                  "MEDIUM": "background-color: #ffa50033",
                  "LOW": "background-color: #00cc6633"}
        if "risk_level" in row.index:
            return [colors.get(row["risk_level"], "")] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(highlight_risk, axis=1).format(
            {"risk_score": "{:.4f}", "response_time_ms": "{:.1f}"}
        ),
        use_container_width=True,
        height=400,
    )

    csv = df_display.to_csv(index=False)
    st.download_button(
        "Download filtered data as CSV",
        csv,
        file_name="commit_risk_predictions.csv",
        mime="text/csv",
    )

elif page == "Live Prediction":
    st.title("Live Commit Risk Prediction")
    st.caption("Enter commit details below and get an instant risk assessment.")

    predictor = get_predictor()
    if predictor is None:
        st.error("Could not load the ML model. Make sure `models/advanced_xgboost.pkl` exists.")
        st.info("Train a model first: `python scripts/run_training.py`")
        st.stop()

    with st.form("predict_form"):
        st.subheader("Commit Details")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            commit_hash = st.text_input("Commit hash", value="manual-test")
            lines_added = st.number_input("Lines added", min_value=0, value=50, step=1)
            lines_deleted = st.number_input("Lines deleted", min_value=0, value=10, step=1)

        with fc2:
            files_changed = st.number_input("Files changed", min_value=1, value=3, step=1)
            touches_core = st.selectbox("Touches core modules?", [0, 1], index=0)
            touches_tests = st.selectbox("Touches test files?", [0, 1], index=0)

        with fc3:
            total_commits = st.number_input("Developer total commits", min_value=1, value=100, step=1)
            buggy_commits = st.number_input("Developer buggy commits", min_value=0, value=10, step=1)
            recent_frequency = st.number_input("Recent commit frequency", min_value=0, value=5, step=1)

        timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        submitted = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)

    if submitted:
        commit_data = {
            "commit_hash": commit_hash,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "files_changed": files_changed,
            "touches_core": touches_core,
            "touches_tests": touches_tests,
            "total_commits": total_commits,
            "buggy_commits": buggy_commits,
            "recent_frequency": recent_frequency,
            "timestamp": timestamp,
        }

        with st.spinner("Running prediction..."):
            result = predictor.predict_commit(commit_data)

        risk_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[result["risk_level"]]

        st.markdown("---")
        st.subheader("Prediction Result")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Risk Score", f"{result['risk_score']:.2%}")
        with rc2:
            st.metric("Risk Level", result["risk_level"])
        with rc3:
            st.metric("Risk Label", "Buggy" if result["risk_label"] == 1 else "Clean")

        st.info(f"**Recommendation:** {result['recommendation']}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["risk_score"] * 100,
            title={"text": "Risk Score"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(0, 204, 102, 0.2)"},
                    {"range": [40, 70], "color": "rgba(255, 165, 0, 0.2)"},
                    {"range": [70, 100], "color": "rgba(255, 75, 75, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "value": result["risk_score"] * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        try:
            from src.monitoring.metrics_collector import MetricsCollector
            collector = MetricsCollector()
            collector.log_prediction(
                commit_hash=result["commit_hash"],
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
                features=commit_data,
                response_time_ms=0.0,
            )
            st.success("Prediction logged. Switch to **Dashboard** to see it in the monitoring view.")
        except Exception:
            pass

    st.markdown("---")
    st.subheader("Quick Presets")
    st.caption("Click a preset to auto-fill the form with example data, then submit.")

    presets = {
        "Small safe commit": {
            "lines_added": 10, "lines_deleted": 3, "files_changed": 1,
            "touches_core": 0, "touches_tests": 1,
            "total_commits": 200, "buggy_commits": 5, "recent_frequency": 3,
        },
        "Large risky commit": {
            "lines_added": 500, "lines_deleted": 200, "files_changed": 15,
            "touches_core": 1, "touches_tests": 0,
            "total_commits": 50, "buggy_commits": 30, "recent_frequency": 12,
        },
        "Medium refactor": {
            "lines_added": 120, "lines_deleted": 80, "files_changed": 6,
            "touches_core": 1, "touches_tests": 1,
            "total_commits": 150, "buggy_commits": 15, "recent_frequency": 7,
        },
    }

    preset_cols = st.columns(len(presets))
    for col, (name, values) in zip(preset_cols, presets.items()):
        with col:
            if st.button(name, use_container_width=True):
                st.session_state["preset"] = values
                st.info(f"Preset **{name}** values: {values}")

elif page == "Model Info":
    st.title("Model Information")

    predictor = get_predictor()
    if predictor is None:
        st.error("Could not load the ML model.")
        st.stop()

    info = predictor.get_model_info()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model Details")
        st.json(info)

    with c2:
        st.subheader("Feature List")
        if "feature_names" in info:
            for i, feat in enumerate(info["feature_names"], 1):
                st.text(f"{i:2d}. {feat}")

    st.markdown("---")
    st.subheader("Feature Importance")

    try:
        model = predictor.model_loader.model
        if hasattr(model, "feature_importances_"):
            feature_names = info.get("feature_names", [f"f{i}" for i in range(len(model.feature_importances_))])
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model type.")
    except Exception as e:
        st.warning(f"Could not extract feature importance: {e}")

    summary_path = Path("models/training_summary.txt")
    if summary_path.exists():
        with st.expander("Training Summary", expanded=False):
            st.code(summary_path.read_text(), language="text")
