import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
ACCENT_COLOR = "#2563eb"
ACCENT_LIGHT = "#eff6ff"
BORDER_COLOR = "#e2e8f0"
TEXT_PRIMARY = "#0f172a"
TEXT_MUTED = "#64748b"

RISK_COLORS = {
    "LOW": {"fg": "#16a34a", "bg": "#f0fdf4", "border": "#bbf7d0"},
    "MEDIUM": {"fg": "#d97706", "bg": "#fffbeb", "border": "#fde68a"},
    "HIGH": {"fg": "#ea580c", "bg": "#fff7ed", "border": "#fed7aa"},
    "CRITICAL": {"fg": "#dc2626", "bg": "#fef2f2", "border": "#fecaca"},
}

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal CSS (Strictly light theme, no emojis, clean typography)
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_PRIMARY};
    }}
    
    /* Clean sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #f8fafc;
        border-right: 1px solid {BORDER_COLOR};
    }}
    
    /* Minimal card container */
    .metric-card {{
        background-color: #ffffff;
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}
    
    .metric-title {{
        font-size: 0.8125rem;
        font-weight: 500;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}
    
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
    }}
    
    .metric-subtitle {{
        font-size: 0.8125rem;
        color: {TEXT_MUTED};
        margin-top: 4px;
    }}

    /* Risk badge */
    .risk-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8125rem;
        letter-spacing: 0.025em;
        text-align: center;
    }}
    
    /* Callout notice */
    .info-callout {{
        background-color: #f8fafc;
        border-left: 3px solid {ACCENT_COLOR};
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-top: 12px;
        font-size: 0.875rem;
        color: {TEXT_MUTED};
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 6px;
        font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Data Fetching & Caching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_backend_health() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def get_schema() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}/schema", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def get_metrics() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data
def load_demo_dataset() -> pd.DataFrame:
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "demo_machines.csv",
        Path("data/demo_machines.csv").resolve(),
        Path("../data/demo_machines.csv").resolve(),
    ]
    for p in candidates:
        if p.is_file():
            return pd.read_csv(p)
    return pd.DataFrame()

def render_risk_badge_html(category: str) -> str:
    palette = RISK_COLORS.get(category, {"fg": "#475569", "bg": "#f1f5f9", "border": "#cbd5e1"})
    return (
        f'<div class="risk-badge" style="color: {palette["fg"]}; background-color: {palette["bg"]}; border: 1px solid {palette["border"]};">'
        f'{category}</div>'
    )

def render_backend_error():
    st.error(
        f"Backend Unavailable: Could not establish a connection to {API_URL}. "
        "Please ensure the FastAPI service is active (`uvicorn main:app --reload --port 8000`)."
    )
    if st.button("Retry connection", key="retry_btn"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid {BORDER_COLOR}; margin-bottom: 20px;">
        <div style="font-weight: 700; font-size: 1.125rem; color: {TEXT_PRIMARY};">Predictive Maintenance</div>
        <div style="font-size: 0.8125rem; color: {TEXT_MUTED};">Asset Health & Reliability</div>
    </div>
    """,
    unsafe_allow_html=True
)

pages = [
    "Single Prediction",
    "Batch Prediction",
    "Analytics",
    "Model Insights"
]

selected_page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

# Backend Status in Sidebar
health = get_backend_health()
if health and health.get("status") == "ok":
    st.sidebar.markdown(
        f"""
        <div style="margin-top: 30px; padding: 12px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; font-size: 0.8125rem; color: #166534;">
            <div style="font-weight: 600;">System Online</div>
            <div style="font-size: 0.75rem; color: #15803d; margin-top: 2px;">Backend: {API_URL}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        f"""
        <div style="margin-top: 30px; padding: 12px; background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; font-size: 0.8125rem; color: #991b1b;">
            <div style="font-weight: 600;">System Offline</div>
            <div style="font-size: 0.75rem; color: #b91c1c; margin-top: 2px;">Target: {API_URL}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Page 1: Single Prediction
# ---------------------------------------------------------------------------
if selected_page == "Single Prediction":
    st.markdown("### Single Asset Prediction")
    st.markdown("Evaluate machine telemetry against trained failure and remaining useful life models.")

    schema = get_schema()
    if not schema:
        render_backend_error()
        st.stop()

    machine_types = schema.get("machine_types", [])
    numeric_schema = schema.get("numeric", {})
    demo_df = load_demo_dataset()

    # Presets handler
    st.markdown("##### Quick Presets")
    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 5])

    def load_preset(target_failure: int):
        if not demo_df.empty:
            match = demo_df[demo_df["Failure_Within_7_Days"] == target_failure]
            if not match.empty:
                row = match.iloc[0].to_dict()
                st.session_state["sp_Machine_Type"] = row.get("Machine_Type") if row.get("Machine_Type") in machine_types else machine_types[0]
                st.session_state["sp_AI_Supervision"] = bool(row.get("AI_Supervision", 0))
                for col_name, bounds in numeric_schema.items():
                    if col_name in ("Machine_Age", "AI_Supervision"):
                        continue
                    val = row.get(col_name)
                    if val is None or pd.isna(val):
                        val = bounds.get("median", 0.0)
                    st.session_state[f"sp_{col_name}"] = float(val)

    with col_btn1:
        if st.button("Load healthy example", use_container_width=True):
            load_preset(target_failure=0)
            st.rerun()

    with col_btn2:
        if st.button("Load high-risk example", use_container_width=True):
            load_preset(target_failure=1)
            st.rerun()

    st.markdown("---")

    # Construct the form
    with st.form("single_prediction_form"):
        st.markdown("##### Telemetry & Specifications")

        # Row 1: Machine Type & Installation Year
        col_type, col_year, col_ai = st.columns([2, 1.5, 1.5])
        with col_type:
            curr_type = st.session_state.get("sp_Machine_Type", machine_types[0] if machine_types else "")
            idx = machine_types.index(curr_type) if curr_type in machine_types else 0
            sel_machine_type = st.selectbox("Machine Type", machine_types, index=idx)

        with col_year:
            year_info = numeric_schema.get("Installation_Year", {"min": 2000.0, "max": 2026.0, "median": 2013.0})
            curr_year = float(st.session_state.get("sp_Installation_Year", year_info["median"]))
            curr_year = max(float(year_info["min"]), min(float(year_info["max"]), curr_year))
            sel_installation_year = st.number_input(
                "Installation Year",
                min_value=float(year_info["min"]),
                max_value=float(year_info["max"]),
                value=curr_year,
                step=1.0,
                format="%.0f"
            )

        with col_ai:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            curr_ai = st.session_state.get("sp_AI_Supervision", False)
            sel_ai_supervision = st.checkbox("AI Supervision Active", value=curr_ai)

        st.markdown("###### Sensor Readings")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:
            t_info = numeric_schema.get("Temperature_C", {"min": -20.0, "max": 130.0, "median": 60.0})
            curr_t = float(st.session_state.get("sp_Temperature_C", t_info["median"]))
            curr_t = max(float(t_info["min"]), min(float(t_info["max"]), curr_t))
            val_temp = st.number_input("Temperature (C)", min_value=float(t_info["min"]), max_value=float(t_info["max"]), value=curr_t, step=0.5, format="%.2f")

        with col_s2:
            v_info = numeric_schema.get("Vibration_mms", {"min": 0.0, "max": 40.0, "median": 10.0})
            curr_v = float(st.session_state.get("sp_Vibration_mms", v_info["median"]))
            curr_v = max(float(v_info["min"]), min(float(v_info["max"]), curr_v))
            val_vib = st.number_input("Vibration (mm/s)", min_value=float(v_info["min"]), max_value=float(v_info["max"]), value=curr_v, step=0.5, format="%.2f")

        with col_s3:
            s_info = numeric_schema.get("Sound_dB", {"min": 20.0, "max": 130.0, "median": 75.0})
            curr_s = float(st.session_state.get("sp_Sound_dB", s_info["median"]))
            curr_s = max(float(s_info["min"]), min(float(s_info["max"]), curr_s))
            val_sound = st.number_input("Sound Level (dB)", min_value=float(s_info["min"]), max_value=float(s_info["max"]), value=curr_s, step=0.5, format="%.2f")

        with col_s4:
            p_info = numeric_schema.get("Power_Consumption_kW", {"min": 0.0, "max": 600.0, "median": 150.0})
            curr_p = float(st.session_state.get("sp_Power_Consumption_kW", p_info["median"]))
            curr_p = max(float(p_info["min"]), min(float(p_info["max"]), curr_p))
            val_power = st.number_input("Power (kW)", min_value=float(p_info["min"]), max_value=float(p_info["max"]), value=curr_p, step=1.0, format="%.2f")

        st.markdown("###### Maintenance & Operating History")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            h_info = numeric_schema.get("Operational_Hours", {"min": 0.0, "max": 100000.0, "median": 50000.0})
            curr_h = float(st.session_state.get("sp_Operational_Hours", h_info["median"]))
            curr_h = max(float(h_info["min"]), min(float(h_info["max"]), curr_h))
            val_hours = st.number_input("Operational Hours", min_value=float(h_info["min"]), max_value=float(h_info["max"]), value=curr_h, step=100.0, format="%.0f")

        with col_m2:
            oil_info = numeric_schema.get("Oil_Level_pct", {"min": 0.0, "max": 100.0, "median": 70.0})
            curr_oil = float(st.session_state.get("sp_Oil_Level_pct", oil_info["median"]))
            curr_oil = max(float(oil_info["min"]), min(float(oil_info["max"]), curr_oil))
            val_oil = st.number_input("Oil Level (%)", min_value=float(oil_info["min"]), max_value=float(oil_info["max"]), value=curr_oil, step=1.0, format="%.2f")

        with col_m3:
            cool_info = numeric_schema.get("Coolant_Level_pct", {"min": 0.0, "max": 100.0, "median": 65.0})
            curr_cool = float(st.session_state.get("sp_Coolant_Level_pct", cool_info["median"]))
            curr_cool = max(float(cool_info["min"]), min(float(cool_info["max"]), curr_cool))
            val_cool = st.number_input("Coolant Level (%)", min_value=float(cool_info["min"]), max_value=float(cool_info["max"]), value=curr_cool, step=1.0, format="%.2f")

        with col_m4:
            days_info = numeric_schema.get("Last_Maintenance_Days_Ago", {"min": 0.0, "max": 365.0, "median": 180.0})
            curr_days = float(st.session_state.get("sp_Last_Maintenance_Days_Ago", days_info["median"]))
            curr_days = max(float(days_info["min"]), min(float(days_info["max"]), curr_days))
            val_days = st.number_input("Last Maint. (Days Ago)", min_value=float(days_info["min"]), max_value=float(days_info["max"]), value=curr_days, step=1.0, format="%.0f")

        col_h1, col_h2, col_h3, _ = st.columns(4)
        with col_h1:
            mh_info = numeric_schema.get("Maintenance_History_Count", {"min": 0.0, "max": 20.0, "median": 5.0})
            curr_mh = float(st.session_state.get("sp_Maintenance_History_Count", mh_info["median"]))
            curr_mh = max(float(mh_info["min"]), min(float(mh_info["max"]), curr_mh))
            val_maint_count = st.number_input("Maintenance Count", min_value=float(mh_info["min"]), max_value=float(mh_info["max"]), value=curr_mh, step=1.0, format="%.0f")

        with col_h2:
            fh_info = numeric_schema.get("Failure_History_Count", {"min": 0.0, "max": 15.0, "median": 2.0})
            curr_fh = float(st.session_state.get("sp_Failure_History_Count", fh_info["median"]))
            curr_fh = max(float(fh_info["min"]), min(float(fh_info["max"]), curr_fh))
            val_fail_count = st.number_input("Past Failures Count", min_value=float(fh_info["min"]), max_value=float(fh_info["max"]), value=curr_fh, step=1.0, format="%.0f")

        with col_h3:
            err_info = numeric_schema.get("Error_Codes_Last_30_Days", {"min": 0.0, "max": 20.0, "median": 3.0})
            curr_err = float(st.session_state.get("sp_Error_Codes_Last_30_Days", err_info["median"]))
            curr_err = max(float(err_info["min"]), min(float(err_info["max"]), curr_err))
            val_err_codes = st.number_input("Error Codes (30 Days)", min_value=float(err_info["min"]), max_value=float(err_info["max"]), value=curr_err, step=1.0, format="%.0f")

        submit_btn = st.form_submit_button("Run Predictive Assessment", use_container_width=True)

    if submit_btn:
        payload = {
            "Machine_Type": sel_machine_type,
            "Installation_Year": float(sel_installation_year),
            "Operational_Hours": float(val_hours),
            "Temperature_C": float(val_temp),
            "Vibration_mms": float(val_vib),
            "Sound_dB": float(val_sound),
            "Oil_Level_pct": float(val_oil),
            "Coolant_Level_pct": float(val_cool),
            "Power_Consumption_kW": float(val_power),
            "Last_Maintenance_Days_Ago": float(val_days),
            "Maintenance_History_Count": float(val_maint_count),
            "Failure_History_Count": float(val_fail_count),
            "AI_Supervision": 1 if sel_ai_supervision else 0,
            "Error_Codes_Last_30_Days": float(val_err_codes),
        }

        with st.spinner("Executing model evaluation..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                if resp.status_code == 200:
                    result = resp.json()
                    st.markdown("#### Assessment Results")

                    col_gauge, col_details = st.columns([1.2, 1.8])
                    prob_val = float(result.get("failure_probability", 0.0))
                    pred_flag = int(result.get("failure_prediction", 0))
                    risk_cat = result.get("risk_category", "LOW")
                    rul_days = result.get("remaining_useful_life_days")

                    with col_gauge:
                        palette = RISK_COLORS.get(risk_cat, {"fg": ACCENT_COLOR})
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob_val,
                            number={"suffix": "%", "font": {"size": 42, "color": TEXT_PRIMARY}},
                            gauge={
                                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                                "bar": {"color": palette["fg"]},
                                "bgcolor": "#f8fafc",
                                "borderwidth": 1,
                                "bordercolor": BORDER_COLOR,
                                "steps": [
                                    {"range": [0, 30], "color": "rgba(22, 163, 74, 0.1)"},
                                    {"range": [30, 60], "color": "rgba(217, 119, 6, 0.1)"},
                                    {"range": [60, 80], "color": "rgba(234, 88, 12, 0.1)"},
                                    {"range": [80, 100], "color": "rgba(220, 38, 38, 0.1)"},
                                ],
                            }
                        ))
                        fig_gauge.update_layout(
                            height=260,
                            margin=dict(l=20, r=20, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font={"family": "Inter, sans-serif"}
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col_details:
                        st.markdown(
                            f"""
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px;">
                                <div class="metric-card">
                                    <div class="metric-title">Risk Category</div>
                                    <div style="margin-top: 6px;">{render_risk_badge_html(risk_cat)}</div>
                                    <div class="metric-subtitle">Threshold: {risk_cat} band</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">7-Day Failure Expected</div>
                                    <div class="metric-value" style="color: {'#dc2626' if pred_flag == 1 else '#16a34a'};">
                                        {'Yes' if pred_flag == 1 else 'No'}
                                    </div>
                                    <div class="metric-subtitle">Binary classification</div>
                                </div>
                                <div class="metric-card" style="grid-column: span 2;">
                                    <div class="metric-title">Estimated Remaining Useful Life (RUL)</div>
                                    <div class="metric-value">
                                        {f"{rul_days:.1f} days" if rul_days is not None else "Not Available"}
                                    </div>
                                    <div class="metric-subtitle">Continuous regression forecast</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.caption("Probabilities are model estimates, not guarantees.")
                else:
                    st.error(f"Prediction failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Page 2: Batch Prediction
# ---------------------------------------------------------------------------
elif selected_page == "Batch Prediction":
    st.markdown("### Batch Asset Prediction")
    st.markdown("Perform fleet-wide predictive assessment via CSV file upload or demo dataset.")

    health = get_backend_health()
    if not health:
        render_backend_error()
        st.stop()

    uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])
    col_demo, _ = st.columns([2, 5])
    use_demo = col_demo.button("Use demo file", use_container_width=True)

    input_df = None
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(input_df)} records from uploaded file.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_demo:
        input_df = load_demo_dataset()
        if not input_df.empty:
            st.info(f"Loaded {len(input_df)} records from demo dataset.")

    if input_df is not None and not input_df.empty:
        st.markdown("##### Uploaded Dataset Preview")
        st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("Execute Batch Prediction", type="primary"):
            with st.spinner("Processing batch predictions with model pipeline..."):
                records = input_df.to_dict(orient="records")
                # Clean NaNs to None for valid JSON serialization
                clean_records = []
                for row in records:
                    clean_records.append({k: (None if pd.isna(v) else v) for k, v in row.items()})

                try:
                    resp = requests.post(f"{API_URL}/predict-batch", json=clean_records, timeout=30)
                    if resp.status_code == 200:
                        predictions = resp.json()
                        pred_df = pd.DataFrame(predictions)

                        results_df = input_df.copy()
                        results_df["Failure_Probability_pct"] = pred_df["failure_probability"]
                        results_df["Failure_Prediction"] = pred_df["failure_prediction"]
                        results_df["Risk_Category"] = pred_df["risk_category"]
                        results_df["Remaining_Useful_Life_days"] = pred_df["remaining_useful_life_days"]

                        total_count = len(results_df)
                        high_crit_count = int(results_df["Risk_Category"].isin(["HIGH", "CRITICAL"]).sum())
                        valid_ruls = results_df["Remaining_Useful_Life_days"].dropna()
                        avg_rul = float(valid_ruls.mean()) if not valid_ruls.empty else 0.0

                        # Summary Cards
                        st.markdown("##### Summary Metrics")
                        col_c1, col_c2, col_c3 = st.columns(3)
                        with col_c1:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">Total Machines Analyzed</div>
                                    <div class="metric-value">{total_count}</div>
                                    <div class="metric-subtitle">Processed records</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        with col_c2:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">High + Critical Risk Count</div>
                                    <div class="metric-value" style="color: {'#dc2626' if high_crit_count > 0 else '#16a34a'};">
                                        {high_crit_count}
                                    </div>
                                    <div class="metric-subtitle">{(high_crit_count/total_count*100):.1f}% of inspected fleet</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        with col_c3:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">Average Estimated RUL</div>
                                    <div class="metric-value">{avg_rul:.1f} days</div>
                                    <div class="metric-subtitle">Fleet mean duration</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Risk Distribution Chart
                        st.markdown("##### Risk Distribution")
                        risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                        risk_counts = results_df["Risk_Category"].value_counts().reindex(risk_order, fill_value=0).reset_index()
                        risk_counts.columns = ["Risk_Category", "Count"]

                        bar_colors = [RISK_COLORS[cat]["fg"] for cat in risk_order]

                        fig_bar = px.bar(
                            risk_counts,
                            x="Risk_Category",
                            y="Count",
                            text="Count",
                            color="Risk_Category",
                            color_discrete_map={cat: RISK_COLORS[cat]["fg"] for cat in risk_order},
                        )
                        fig_bar.update_layout(
                            plot_bgcolor="#ffffff",
                            paper_bgcolor="#ffffff",
                            height=320,
                            margin=dict(l=20, r=20, t=20, b=20),
                            showlegend=False,
                            xaxis=dict(title="Risk Category", linecolor=BORDER_COLOR),
                            yaxis=dict(title="Count", linecolor=BORDER_COLOR, gridcolor="#f1f5f9"),
                            font=dict(family="Inter, sans-serif")
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Results Table
                        st.markdown("##### Detailed Prediction Results")
                        display_cols = [
                            "Machine_Type",
                            "Risk_Category",
                            "Failure_Probability_pct",
                            "Failure_Prediction",
                            "Remaining_Useful_Life_days",
                            "Operational_Hours",
                            "Temperature_C",
                            "Vibration_mms"
                        ]
                        existing_cols = [c for c in display_cols if c in results_df.columns]
                        other_cols = [c for c in results_df.columns if c not in existing_cols]
                        reordered_df = results_df[existing_cols + other_cols]

                        st.dataframe(reordered_df, use_container_width=True)

                        # Download button
                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Results CSV",
                            data=csv_bytes,
                            file_name="predictive_maintenance_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"Batch prediction failed with status {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Batch request error: {e}")

# ---------------------------------------------------------------------------
# Page 3: Analytics
# ---------------------------------------------------------------------------
elif selected_page == "Analytics":
    st.markdown("### Fleet Telemetry Analytics")
    st.markdown("Exploratory analysis of sensor distributions, correlations, and historical failure patterns.")

    df = load_demo_dataset()
    if df.empty:
        st.warning("Historical demo dataset not found in data/demo_machines.csv.")
        st.stop()

    st.markdown(f"Analyzed sample size: **{len(df)} machines**")

    # 1. Failure Rate by Machine_Type
    st.markdown("##### Failure Rate by Machine Type")
    fail_rate = df.groupby("Machine_Type")["Failure_Within_7_Days"].agg(["mean", "count"]).reset_index()
    fail_rate["failure_rate_pct"] = (fail_rate["mean"] * 100).round(1)
    fail_rate = fail_rate.sort_values(by="failure_rate_pct", ascending=True)

    fig_type = px.bar(
        fail_rate,
        x="failure_rate_pct",
        y="Machine_Type",
        orientation="h",
        labels={"failure_rate_pct": "Failure Rate (%)", "Machine_Type": "Machine Type"},
        color_discrete_sequence=[ACCENT_COLOR],
        height=600,
    )
    fig_type.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Failure Rate (%)", gridcolor="#f1f5f9", linecolor=BORDER_COLOR),
        yaxis=dict(title="", linecolor=BORDER_COLOR),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_type, use_container_width=True)

    # 2. Temperature vs Vibration Scatter
    st.markdown("##### Temperature vs Vibration Coloured by Failure")
    scatter_df = df.copy()
    scatter_df["Status"] = scatter_df["Failure_Within_7_Days"].map({0: "Normal", 1: "Failure Imminent"})

    fig_scatter = px.scatter(
        scatter_df,
        x="Temperature_C",
        y="Vibration_mms",
        color="Status",
        color_discrete_map={"Normal": "#94a3b8", "Failure Imminent": "#dc2626"},
        hover_data=["Machine_Type", "Operational_Hours", "Error_Codes_Last_30_Days"],
        labels={"Temperature_C": "Temperature (C)", "Vibration_mms": "Vibration (mm/s)"},
        height=450,
    )
    fig_scatter.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Temperature (C)", gridcolor="#f1f5f9", linecolor=BORDER_COLOR),
        yaxis=dict(title="Vibration (mm/s)", gridcolor="#f1f5f9", linecolor=BORDER_COLOR),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Boxplots of Vibration, Temperature, and Error Codes
    st.markdown("##### Feature Distributions by Failure Status")
    box_cols = ["Vibration_mms", "Temperature_C", "Error_Codes_Last_30_Days"]
    titles = ["Vibration (mm/s)", "Temperature (C)", "Error Codes (Last 30 Days)"]

    fig_box = make_subplots(rows=1, cols=3, subplot_titles=titles)
    for i, col in enumerate(box_cols, start=1):
        for status_val, label, color in [(0, "Normal", "#94a3b8"), (1, "Failure", "#dc2626")]:
            sub = df[df["Failure_Within_7_Days"] == status_val]
            fig_box.add_trace(
                go.Box(
                    y=sub[col],
                    name=label,
                    marker_color=color,
                    showlegend=(i == 1),
                    boxmean=True
                ),
                row=1, col=i
            )

    fig_box.update_layout(
        height=380,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif")
    )
    fig_box.update_xaxes(showline=True, linecolor=BORDER_COLOR)
    fig_box.update_yaxes(showline=True, linecolor=BORDER_COLOR, gridcolor="#f1f5f9")
    st.plotly_chart(fig_box, use_container_width=True)

    # 4. Correlation Heatmap
    st.markdown("##### Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)

    fig_heat = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Correlation"),
        height=650,
    )
    fig_heat.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 4: Model Insights
# ---------------------------------------------------------------------------
elif selected_page == "Model Insights":
    st.markdown("### Model Insights & Evaluation")
    st.markdown("Performance metrics, explainability analysis, and training methodology.")

    metrics = get_metrics()
    if not metrics:
        render_backend_error()
        st.stop()

    st.markdown("##### Key Performance Metrics")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    # Precision & Recall
    with col_m1:
        recall_val = metrics.get("recall")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Recall</div>
                <div class="metric-value">{f"{recall_val:.4f}" if recall_val is not None else "N/A"}</div>
                <div class="metric-subtitle">Sensitivity to failures</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m2:
        pr_auc = metrics.get("pr_auc")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">PR-AUC</div>
                <div class="metric-value">{f"{pr_auc:.4f}" if pr_auc is not None else "N/A"}</div>
                <div class="metric-subtitle">Precision-Recall Area</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m3:
        f1_val = metrics.get("f1")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value">{f"{f1_val:.4f}" if f1_val is not None else "N/A"}</div>
                <div class="metric-subtitle">Harmonic mean</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m4:
        roc_auc = metrics.get("roc_auc")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">ROC-AUC</div>
                <div class="metric-value">{f"{roc_auc:.4f}" if roc_auc is not None else "N/A"}</div>
                <div class="metric-subtitle">Receiver Operating Char.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Additional metrics table for everything present
    st.markdown("##### All Model Metrics")
    metrics_table = [
        {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
        for k, v in metrics.items()
    ]
    st.table(pd.DataFrame(metrics_table))

    # SHAP Summary Plot
    st.markdown("##### Feature Importance & Explainability")
    shap_paths = [
        Path(__file__).resolve().parent.parent / "models" / "shap_summary.png",
        Path("models/shap_summary.png").resolve(),
        Path("../models/shap_summary.png").resolve(),
    ]
    shap_found = False
    for p in shap_paths:
        if p.is_file():
            st.image(str(p), caption="SHAP Summary Plot (Feature Impact on Failure Predictions)", use_container_width=True)
            shap_found = True
            break

    if not shap_found:
        st.info("SHAP summary plot (shap_summary.png) not found in models directory.")

    # Methodology Note
    st.markdown("##### Methodology & Design Rationale")
    st.markdown(
        """
        <div class="info-callout">
            <strong>Imbalance Handling:</strong> Predictive maintenance datasets suffer from severe class imbalance,
            with machine failures representing a rare positive class. The classifier utilizes positive instance weighting 
            (<code>scale_pos_weight</code>) during training to penalize uncaptured failures.<br><br>
            <strong>Evaluation Criteria:</strong> In mission-critical industrial asset monitoring, <strong>Recall</strong> 
            and <strong>PR-AUC</strong> are prioritized over standard accuracy. An undetected catastrophic failure introduces 
            unacceptable equipment downtime and severe replacement expenses, whereas false positives represent minor inspection costs.
        </div>
        """,
        unsafe_allow_html=True
    )
