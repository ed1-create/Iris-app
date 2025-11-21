"""
Professional Streamlit Dashboard Application
Combining Patient Segmentation and Drug Adherence Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import utility functions
from utils.segmentation_utils import (
    load_segmentation_data,
    get_cluster_summary,
    get_patient_cluster,          # still available if you need it elsewhere
    get_evaluation_metrics,
    get_visualization_path,
    get_available_visualizations,
    create_cluster_comparison_data,
    get_patient_details,
    compare_patient_to_cluster,
)
from utils.adherence_utils import (
    load_adherence_data,
    create_ts_features,
    train_xgboost_model,
    train_arima_model,
    calculate_metrics,
    generate_forecast,
    get_adherence_status,
)

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# THEME: SEMI-DARK, TABLEAU / POWERBI STYLE
# -----------------------------------------------------------------------------
BG_APP = "#020617"          # almost-black navy
BG_CONTENT = "#020617"
CARD_BG = "#0b1220"         # dark card
CARD_BG_SOFT = "#020617"
ACCENT_PRIMARY = "#38bdf8"  # sky blue
ACCENT_SECONDARY = "#22c55e"  # green
ACCENT_WARNING = "#f97316"  # orange
ACCENT_DANGER = "#ef4444"   # red
TEXT_PRIMARY = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
BORDER_COLOR = "#1f2937"

# Global CSS
st.markdown(
    f"""
<style>
    /* App background */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {BG_APP} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stHeader"] {{
        background: linear-gradient(90deg, #020617 0%, #020617 100%) !important;
    }}

    /* Main title and subtitle */
    .main-header {{
        font-size: 34px;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        text-align: left;
        padding: 4px 0 0 0;
        margin-bottom: 0;
    }}
    .subtitle {{
        text-align: left;
        color: {TEXT_MUTED};
        font-size: 14px;
        margin-bottom: 12px;
        font-weight: 400;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
    }}
    p, span, li, label, strong, div {{
        color: {TEXT_PRIMARY};
    }}

    /* Horizontal separators */
    hr {{
        border: none;
        border-top: 1px solid {BORDER_COLOR};
        margin: 0.75rem 0;
    }}

    /* Cards / info boxes */
    .info-box {{
        background-color: {CARD_BG};
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        margin: 12px 0;
        color: {TEXT_PRIMARY};
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
    }}
    .success-box {{
        background-color: #022c22;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #16a34a;
        margin: 12px 0;
        color: {TEXT_PRIMARY};
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
    }}
    .warning-box {{
        background-color: #451a03;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid {ACCENT_WARNING};
        margin: 12px 0;
        color: {TEXT_PRIMARY};
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
    }}

    .info-box h3, .info-box h4, .info-box p, .info-box li,
    .success-box h3, .success-box h4, .success-box p, .success-box li,
    .warning-box h3, .warning-box h4, .warning-box p, .warning-box li {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(90deg, {ACCENT_PRIMARY}, #0ea5e9);
        color: #0b1120 !important;
        border: none;
        padding: 0.5rem 1.3rem;
        font-size: 0.92rem;
        font-weight: 600;
        border-radius: 999px;
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.45);
        transition: all 0.15s ease-out;
    }}
    .stButton>button:hover {{
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 15px 35px rgba(56, 189, 248, 0.7);
        cursor: pointer;
    }}

    /* Metric labels & values */
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.8rem;
    }}
    [data-testid="stMetricValue"] {{
        color: {ACCENT_PRIMARY} !important;
        font-weight: 700 !important;
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #020617 !important;
        border-right: 1px solid {BORDER_COLOR};
        box-shadow: 4px 0 16px rgba(0,0,0,0.7);
    }}

    .sidebar-header {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {TEXT_MUTED};
        margin-top: 0.65rem;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }}

    .sidebar-caption {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-bottom: 0.8rem;
    }}

    /* Sidebar nav radio as nav pills */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.55rem;
        border-radius: 999px;
        margin-bottom: 0.2rem;
        font-size: 0.88rem;
        color: {TEXT_PRIMARY};
        border: 1px solid transparent;
    }}
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {{
        background-color: rgba(30, 64, 175, 0.35);
        border-color: rgba(59, 130, 246, 0.8);
    }}

    /* Sub-level radio indentation */
    .sidebar-sub-radio > div[role="radiogroup"] > label {{
        margin-left: 0.5rem;
        padding-left: 0.9rem;
        font-size: 0.85rem;
        opacity: 0.95;
    }}

    .footer-text {{
        color: {TEXT_MUTED};
        font-size: 12px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown(
        '<p class="main-header">🏥 Healthcare Analytics Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Semi-dark analytics workspace • Patient Segmentation & Drug Adherence Forecasting</p>',
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown(
        """
        <div style="display:flex; justify-content:flex-end; align-items:center; height:100%;">
            <div style="padding:8px 14px; border-radius:999px; border:1px solid #1f2937; background:#020617; font-size:11px; color:#9ca3af;">
                <span style="color:#38bdf8;">●</span> Connected · Analytics environment
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (ICON-BASED)
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.35rem; margin-bottom:0.6rem;">
        <span style="font-size:1.15rem;">📊</span>
        <span style="font-weight:600; color:#e5e7eb;">Analytics Navigation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "section" not in st.session_state:
    st.session_state["section"] = "segmentation"

cohort = None
time_period = None

# Top-level area
st.sidebar.markdown(
    '<div class="sidebar-header">Workspace</div>', unsafe_allow_html=True
)
main_section = st.sidebar.radio(
    "Select analytics area",
    ["👥 Patient Segmentation", "💊 Drug Adherence Prediction", "💊 Drug Consumption Forecasting"],
    index=0 if st.session_state["section"] == "segmentation" else (1 if st.session_state["section"] == "adherence" else 2),
    label_visibility="collapsed",
)

st.sidebar.markdown("<hr style='margin:0.5rem 0;'/>", unsafe_allow_html=True)

# Submenus
if main_section.startswith("👥"):
    st.session_state["section"] = "segmentation"

    st.sidebar.markdown(
        '<div class="sidebar-header">Segmentation cohorts</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="sidebar-caption">Choose which diagnostic group you want to analyze.</div>',
        unsafe_allow_html=True,
    )

    sub_segment = st.sidebar.radio(
        "Segmentation cohort",
        ["🩺 I10 – Hypertension", "🧪 Z01 – Preventive care"],
        key="segmentation_cohort_radio",
        label_visibility="collapsed",
    )
    cohort = "i10" if sub_segment.startswith("🩺") else "z01"
    st.session_state["cohort"] = cohort

elif main_section.startswith("💊") and "Consumption" in main_section:
    st.session_state["section"] = "consumption"

else:
    st.session_state["section"] = "adherence"

    st.sidebar.markdown(
        '<div class="sidebar-header">Adherence resolution</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="sidebar-caption">Select the time granularity for forecasting.</div>',
        unsafe_allow_html=True,
    )

    sub_segment = st.sidebar.radio(
        "Adherence granularity",
        ["📅 Monthly adherence", "⏱️ Bi-weekly adherence"],
        key="adherence_granularity_radio",
        label_visibility="collapsed",
    )
    time_period = "Monthly" if sub_segment.startswith("📅") else "Bi-weekly"

st.sidebar.markdown("<hr style='margin-top:0.9rem;'/>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS (defined outside conditional to ensure proper caching)
# -----------------------------------------------------------------------------
def load_cohort_data(_cohort: str):
    """Load segmentation data for a specific cohort (no caching to avoid cache issues)"""
    try:
        # Normalize cohort to ensure consistent file loading
        _cohort = _cohort.lower().strip()
        
        # Load data directly without going through cached utility function
        from pathlib import Path
        import pandas as pd
        
        base_path = Path("notebooks/outputs/data")
        cohort_lower = _cohort.lower()
        
        data = {}
        
        # Load each file directly to ensure we get the right cohort's data
        data['profiles'] = pd.read_csv(base_path / f"{cohort_lower}_cluster_profiles.csv")
        data['assignments'] = pd.read_csv(base_path / f"{cohort_lower}_cluster_assignments.csv")
        data['evaluation'] = pd.read_csv(base_path / f"{cohort_lower}_clustering_evaluation.csv")
        data['medoids'] = pd.read_csv(base_path / f"{cohort_lower}_cluster_medoids.csv")
        
        return data
    except Exception as e:
        import sys
        print(f"Error loading data for cohort {_cohort}: {e}", file=sys.stderr)
        return None

def get_metrics(_cohort: str):
    """Get evaluation metrics for a specific cohort (no caching to avoid cache issues)
    
    Args:
        _cohort: Cohort identifier ('i10' or 'z01')
    """
    try:
        # Normalize cohort to ensure consistent file loading
        _cohort = _cohort.lower().strip()
        
        # Load metrics directly - no caching to ensure correct data
        from pathlib import Path
        import pandas as pd
        
        base_path = Path("notebooks/outputs/data")
        eval_file = base_path / f"{_cohort}_clustering_evaluation.csv"
        
        if not eval_file.exists():
            raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
        
        eval_df = pd.read_csv(eval_file)
        
        if len(eval_df) == 0:
            raise ValueError(f"Evaluation file is empty: {eval_file}")
        
        return {
            'optimal_k': int(eval_df['optimal_k'].iloc[0]),
            'silhouette_score': float(eval_df['silhouette_score'].iloc[0]),
            'stability_jaccard_mean': float(eval_df['stability_jaccard_mean'].iloc[0]),
            'n_patients': int(eval_df['n_patients'].iloc[0])
        }
    except Exception as e:
        import sys
        print(f"Error loading metrics for cohort {_cohort}: {e}", file=sys.stderr)
        return None

# -----------------------------------------------------------------------------
# PATIENT SEGMENTATION MODULE
# -----------------------------------------------------------------------------
if st.session_state["section"] == "segmentation":
    st.header("👥 Patient Segmentation Workspace")

    # Get cohort from session state (set by sidebar)
    cohort = st.session_state.get("cohort", "i10")
    
    # Ensure cohort is lowercase for consistency
    cohort = cohort.lower().strip()
    
    cohort_display = "I10 – Hypertension" if cohort == "i10" else "Z01 – Preventive care"

    st.subheader(f"🏥 Active Cohort: {cohort_display}")

    st.markdown(
        f"""
        <div class="info-box">
            <h3 style="margin-bottom:4px;">Cohort scope</h3>
            <p style="margin-bottom:0;">All metrics, clusters, and visualizations on this page are computed for <strong>{cohort_display}</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------- Data loading --------
    # Clear any potential cache issues by ensuring we're using the correct cohort
    data = load_cohort_data(cohort)

    if data is None:
        st.error(f"Failed to load segmentation data for {cohort_display}. Please check data files.")
        st.stop()

    # Verify we have the correct data
    if "profiles" not in data or data["profiles"] is None or len(data["profiles"]) == 0:
        st.error(f"No profile data found for {cohort_display}.")
        st.stop()

    # Load metrics directly (no caching to avoid cache issues)
    metrics = get_metrics(cohort)

    if metrics is None:
        st.error("Failed to load evaluation metrics.")
        st.stop()

    # -------- Executive summary --------
    st.subheader("📊 Cohort Summary Panel")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total patients",
            f"{metrics['n_patients']:,}",
            help="Total number of patients included in the cohort",
        )

    with col2:
        st.metric(
            "Optimal clusters (k)",
            f"{metrics['optimal_k']}",
            help="Number of data-driven patient segments identified",
        )

    with col3:
        st.metric(
            "Silhouette score",
            f"{metrics['silhouette_score']:.3f}",
            help="Internal cluster quality metric (higher is better)",
        )

    with col4:
        st.metric(
            "Stability (Jaccard)",
            f"{metrics['stability_jaccard_mean']:.3f}",
            help="Bootstrap-based cluster stability index",
        )

    with col5:
        avg_cluster_size = metrics["n_patients"] / metrics["optimal_k"]
        st.metric(
            "Average cluster size",
            f"{avg_cluster_size:.0f}",
            help="Average number of patients per segment",
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Cluster overview --------
    st.subheader("📈 Cluster Distribution & Profiles")

    # Ensure we're using the correct cohort's data
    # Get fresh copy to avoid any potential reference issues
    profiles = data["profiles"].copy()
    
    # Verify we have data for this cohort
    if profiles.empty:
        st.error(f"No cluster profile data available for {cohort_display}.")
        st.stop()
    
    # Sort by cluster ID to ensure consistent ordering
    profiles = profiles.sort_values("cluster").reset_index(drop=True)
    
    # Debug output (uncomment to verify data is changing)
    # st.write(f"**Debug Info:** Cohort={cohort}, Clusters={len(profiles)}, Total Patients={profiles['n_patients'].sum()}")

    col1, col2 = st.columns(2)

    with col1:
        # Interactive bar chart with Plotly
        cluster_sizes = profiles["n_patients"].values
        cluster_labels = [f"Cluster {int(i)}" for i in profiles["cluster"].values]
        colors_professional = [
            "#38bdf8",
            "#22c55e",
            "#f97316",
            "#e879f9",
            "#a855f7",
            "#06b6d4",
        ]
        bar_colors = [
            colors_professional[i % len(colors_professional)]
            for i in range(len(cluster_sizes))
        ]

        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=cluster_labels,
                    y=cluster_sizes,
                    marker=dict(
            color=bar_colors,
                        line=dict(color="#111827", width=1.5),
                    ),
                    text=[
                        f"{int(size):,}<br>({pct:.1f}%)"
                        for size, pct in zip(
                            cluster_sizes, profiles["pct_patients"].values
                        )
                    ],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Patients: %{y:,}<br>Percentage: %{customdata:.1f}%<extra></extra>",
                    customdata=profiles["pct_patients"].values,
                )
            ]
        )

        fig_bar.update_layout(
            title=dict(
                text=f"<b>Cluster Size Distribution — {cohort_display}</b>",
                font=dict(size=16, color=TEXT_PRIMARY),
                x=0.5,
            ),
            xaxis=dict(
                title=dict(text="<b>Cluster</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                tickfont=dict(size=11, color=TEXT_PRIMARY),
                gridcolor=BORDER_COLOR,
                gridwidth=1,
            ),
            yaxis=dict(
                title=dict(
                    text="<b>Number of Patients</b>", font=dict(size=12, color=TEXT_PRIMARY)
                ),
                tickfont=dict(size=11, color=TEXT_PRIMARY),
                gridcolor=BORDER_COLOR,
                gridwidth=1,
            ),
            plot_bgcolor=BG_CONTENT,
            paper_bgcolor=BG_CONTENT,
            font=dict(color=TEXT_PRIMARY),
            hovermode="closest",
            height=450,
            margin=dict(l=50, r=50, t=60, b=50),
            )

        st.plotly_chart(
            fig_bar, 
            use_container_width=True, 
            config={"displayModeBar": True}
        )

    with col2:
        # Interactive pie chart with Plotly
        colors_professional = [
            "#38bdf8",
            "#22c55e",
            "#f97316",
            "#e879f9",
            "#a855f7",
            "#06b6d4",
        ]
        colors_pie = [
            colors_professional[i % len(colors_professional)]
            for i in range(len(profiles))
        ]

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=[f"Cluster {int(i)}" for i in profiles["cluster"].values],
                    values=profiles["pct_patients"].values,
                    marker=dict(colors=colors_pie, line=dict(color="#111827", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=11, color=TEXT_PRIMARY),
                    hovertemplate="<b>%{label}</b><br>Percentage: %{percent}<br>Patients: %{customdata:,}<extra></extra>",
                    customdata=profiles["n_patients"].values,
                    hole=0.3,  # Makes it a donut chart for better aesthetics
                )
            ]
        )

        fig_pie.update_layout(
            title=dict(
                text=f"<b>Share of Cohort per Cluster — {cohort_display}</b>",
                font=dict(size=16, color=TEXT_PRIMARY),
                x=0.5,
            ),
            plot_bgcolor=BG_CONTENT,
            paper_bgcolor=BG_CONTENT,
            font=dict(color=TEXT_PRIMARY),
            height=450,
            margin=dict(l=50, r=50, t=60, b=50),
            showlegend=True,
            legend=dict(
                font=dict(size=11, color=TEXT_PRIMARY),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER_COLOR,
                borderwidth=1,
            ),
        )

        st.plotly_chart(
            fig_pie, 
            use_container_width=True, 
            config={"displayModeBar": True}
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Cluster explorer (accordion grid) --------
    st.subheader("🔍 Cluster Profile Explorer")

    cluster_ids = profiles["cluster"].tolist()

    def render_cluster_card(cluster_id: int):
        """Renders one accordion card for a cluster in the grid."""
        cluster_row = profiles[profiles["cluster"] == cluster_id].iloc[0]
        cluster_name = cluster_row["cluster_name"]
        cluster_size = int(cluster_row["n_patients"])
        cluster_pct = float(cluster_row["pct_patients"])

        with st.expander(f"Cluster {cluster_id} — {cluster_name}", expanded=False):
            # High-level info
            st.markdown(
                f"""
                <div class="info-box" style="margin-top:0;">
                    <h4 style="margin-bottom:4px;">Cluster overview</h4>
                    <p style="margin-bottom:2px;"><strong>Cluster ID:</strong> {cluster_id}</p>
                    <p style="margin-bottom:0;"><strong>Size:</strong> {cluster_size:,} patients ({cluster_pct:.1f}%)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Key clinical indicators
            cluster_data = get_cluster_summary(cohort, cluster_id)
            st.markdown("**Key clinical indicators**")

            c1, c2, c3 = st.columns(3)

            with c1:
                if "age_median" in cluster_data and cluster_data["age_median"] is not None:
                    st.metric("Median age", f"{cluster_data['age_median']:.0f} yrs")
                if (
                    "encounter_count_12m_median" in cluster_data
                    and cluster_data["encounter_count_12m_median"] is not None
                ):
                    st.metric(
                        "Encounters / year",
                        f"{cluster_data['encounter_count_12m_median']:.0f}",
                    )

            with c2:
                if (
                    "sbp_latest_median" in cluster_data
                    and cluster_data["sbp_latest_median"] is not None
                ):
                    st.metric("Median SBP", f"{cluster_data['sbp_latest_median']:.0f} mmHg")
                if (
                    "icd3_count_median" in cluster_data
                    and cluster_data["icd3_count_median"] is not None
                ):
                    st.metric(
                        "ICD-3 codes (median)",
                        f"{cluster_data['icd3_count_median']:.0f}",
                    )

            with c3:
                # BMI depending on cohort
                if cohort == "z01" and "bmi_category" in cluster_data:
                    st.metric("BMI category", cluster_data["bmi_category"])
                elif (
                    "bmi_latest_median" in cluster_data
                    and cluster_data["bmi_latest_median"] is not None
                ):
                    st.metric(
                        "Median BMI",
                        f"{cluster_data['bmi_latest_median']:.1f} kg/m²",
                    )

            # Optional quick comorbidity look
            comorbidity_cols = [
                col for col in cluster_data.keys() if "has_" in col and "_pct" in col
            ]
            if comorbidity_cols:
                st.markdown("**Top comorbidities (by prevalence)**")
                comorbidity_data = {
                    col.replace("has_", "").replace("_pct", ""): cluster_data[col]
                    for col in comorbidity_cols
                    if cluster_data[col] is not None
                }
                if comorbidity_data:
                    top_comorbid = sorted(
                        comorbidity_data.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    for name, pct in top_comorbid:
                        st.write(f"- **{name}**: {pct:.1f}%")

    # Render accordions in grid (2 per row)
    cols_per_row = 2
    for i in range(0, len(cluster_ids), cols_per_row):
        row_ids = cluster_ids[i : i + cols_per_row]
        row_cols = st.columns(len(row_ids))
        for col_obj, cid in zip(row_cols, row_ids):
            with col_obj:
                render_cluster_card(cid)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Interactive Disease Prevalence Dashboard --------
    st.subheader("🖼️ Disease Prevalence Dashboard")

    # Get all cluster IDs
    cluster_ids = sorted(profiles["cluster"].tolist())
    
    st.markdown(
        """
        <div class="info-box" style="margin-bottom:1rem;">
            <h4 style="margin-bottom:4px;">Select Cluster</h4>
            <p style="margin-bottom:0; font-size:0.9rem;">Choose a cluster to view disease prevalence and comorbidity patterns.</p>
        </div>
        """,
                        unsafe_allow_html=True,
                    )
    
    # Create radio buttons for cluster selection
    def format_cluster_label(cluster_id):
        """Format cluster label for radio button"""
        cluster_row = profiles[profiles['cluster'] == cluster_id]
        if len(cluster_row) > 0 and 'cluster_name' in cluster_row.columns:
            cluster_name = cluster_row['cluster_name'].iloc[0]
            return f"Cluster {cluster_id} — {cluster_name}"
        return f"Cluster {cluster_id}"
    
    selected_cluster = st.radio(
        "Cluster Selection",
        options=cluster_ids,
        format_func=format_cluster_label,
        horizontal=True,
        key=f"disease_prevalence_cluster_{cohort}",
                        )

    # Get cluster data for selected cluster
    cluster_data = get_cluster_summary(cohort, selected_cluster)
    
    if cluster_data:
        # Extract disease/comorbidity columns (pattern: has_*_pct)
        comorbidity_cols = [
            col for col in cluster_data.keys() 
            if "has_" in col and "_pct" in col and cluster_data[col] is not None
        ]
        
        if comorbidity_cols:
            # Prepare data for visualization
            disease_data = []
            for col in comorbidity_cols:
                disease_name = col.replace("has_", "").replace("_pct", "").replace("_", " ").title()
                prevalence = float(cluster_data[col])
                disease_data.append({
                    "Disease": disease_name,
                    "Prevalence (%)": prevalence,
                    "ICD Code": col.replace("has_", "").replace("_pct", "").upper()
                })
            
            # Sort by prevalence (descending)
            disease_data = sorted(disease_data, key=lambda x: x["Prevalence (%)"], reverse=True)
            disease_df = pd.DataFrame(disease_data)
            
            # Get cluster name for chart title
            cluster_name = cluster_data.get("cluster_name", f"Cluster {selected_cluster}")
            
            # Interactive bar chart
            colors_professional = [
                "#38bdf8",
                "#22c55e",
                "#f97316",
                "#e879f9",
                "#a855f7",
                "#06b6d4",
            ]
            
            # Create color mapping based on prevalence (gradient)
            bar_colors = []
            max_prev = disease_df["Prevalence (%)"].max()
            min_prev = disease_df["Prevalence (%)"].min()
            
            for prev in disease_df["Prevalence (%)"]:
                if max_prev > min_prev:
                    # Normalize to 0-1 range
                    normalized = (prev - min_prev) / (max_prev - min_prev)
                    # Use gradient from blue (low) to orange (high)
                    if normalized < 0.5:
                        # Blue to green gradient
                        r = int(56 + (34 * normalized * 2))
                        g = int(189 + (66 * normalized * 2))
                        b = 248
                    else:
                        # Green to orange gradient
                        r = int(34 + (249 * (normalized - 0.5) * 2))
                        g = int(197 + (115 * (normalized - 0.5) * 2))
                        b = int(248 - (22 * (normalized - 0.5) * 2))
                    bar_colors.append(f"rgb({r}, {g}, {b})")
                else:
                    bar_colors.append(colors_professional[0])
            
            fig_disease = go.Figure(
                data=[
                    go.Bar(
                        x=disease_df["Disease"],
                        y=disease_df["Prevalence (%)"],
                        marker=dict(
                            color=bar_colors,
                            line=dict(color="#111827", width=1.5),
                        ),
                        text=[
                            f"{prev:.1f}%"
                            for prev in disease_df["Prevalence (%)"]
                        ],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Prevalence: %{y:.2f}%<br>ICD: %{customdata}<extra></extra>",
                        customdata=disease_df["ICD Code"],
                    )
                ]
            )
            
            fig_disease.update_layout(
                title=dict(
                    text=f"<b>Disease Prevalence — {cluster_name}</b>",
                    font=dict(size=18, color=TEXT_PRIMARY),
                    x=0.5,
                ),
                xaxis=dict(
                    title=dict(text="<b>Disease/Condition</b>", font=dict(size=13, color=TEXT_PRIMARY)),
                    tickfont=dict(size=11, color=TEXT_PRIMARY),
                    gridcolor=BORDER_COLOR,
                    gridwidth=1,
                ),
                yaxis=dict(
                    title=dict(text="<b>Prevalence (%)</b>", font=dict(size=13, color=TEXT_PRIMARY)),
                    tickfont=dict(size=11, color=TEXT_PRIMARY),
                    gridcolor=BORDER_COLOR,
                    gridwidth=1,
                    range=[0, max(disease_df["Prevalence (%)"].max() * 1.1, 10)],
                ),
                plot_bgcolor=BG_CONTENT,
                paper_bgcolor=BG_CONTENT,
                font=dict(color=TEXT_PRIMARY),
                hovermode="closest",
                height=550,
                margin=dict(l=50, r=50, t=80, b=150),
            )
            
            # Rotate x-axis labels for better readability
            fig_disease.update_xaxes(tickangle=-45)
            
            st.plotly_chart(fig_disease, use_container_width=True, config={"displayModeBar": True})
    else:
            st.warning(
                f"No disease/comorbidity data available for Cluster {selected_cluster}. "
                "Please ensure the cluster profiles include disease prevalence columns (has_*_pct)."
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Clinical Profile Heatmap Dashboard --------
    st.subheader("🔥 Clinical Profile Heatmap")
    
    # Prepare data for heatmap - matching the image structure
    # Select key clinical features for heatmap (in order shown in image)
    feature_cols = []
    
    # Core features (in order from image)
    if "sbp_latest_median" in profiles.columns:
        feature_cols.append(("sbp_latest_median", "SBP (mmHg)"))
    if "dbp_latest_median" in profiles.columns:
        feature_cols.append(("dbp_latest_median", "DBP (mmHg)"))
    if "bmi_latest_median" in profiles.columns:
        feature_cols.append(("bmi_latest_median", "BMI"))
    if "age_median" in profiles.columns:
        feature_cols.append(("age_median", "Age (years)"))
    if "encounter_count_12m_median" in profiles.columns:
        feature_cols.append(("encounter_count_12m_median", "Encounters (12m)"))
    if "icd3_count_median" in profiles.columns:
        feature_cols.append(("icd3_count_median", "ICD3 Count"))
    
    # Disease prevalence features
    if cohort == "i10":
        if "has_E78_pct" in profiles.columns:
            feature_cols.append(("has_E78_pct", "Dyslipidemia (%)"))
        if "has_I70_pct" in profiles.columns:
            feature_cols.append(("has_I70_pct", "Atherosclerosis (%)"))
    elif cohort == "z01":
        if "has_I10_pct" in profiles.columns:
            feature_cols.append(("has_I10_pct", "Hypertension (%)"))
        if "has_E78_pct" in profiles.columns:
            feature_cols.append(("has_E78_pct", "Dyslipidemia (%)"))
    
    if feature_cols:
        # Create heatmap data
        heatmap_data = []
        cluster_labels = []
        feature_labels = []
        
        for cluster_id in sorted(profiles["cluster"].values):
            cluster_row = profiles[profiles["cluster"] == cluster_id].iloc[0]
            cluster_labels.append(f"C{int(cluster_id)}")  # Use C0, C1, C2, etc. format
            
            row_data = []
            for col_name, display_name in feature_cols:
                if col_name in cluster_row and pd.notna(cluster_row[col_name]):
                    row_data.append(float(cluster_row[col_name]))
                else:
                    row_data.append(0.0)
            
            heatmap_data.append(row_data)
        
        # Extract feature labels
        feature_labels = [display_name for _, display_name in feature_cols]
        
        # Convert to numpy array for processing
        import numpy as np
        heatmap_array = np.array(heatmap_data)
        
        # Normalize each row (feature) independently using min-max normalization
        # This allows comparison of relative differences within each feature
        normalized_by_row = []
        for row_idx in range(heatmap_array.shape[0]):  # For each cluster
            normalized_row = []
            for col_idx in range(heatmap_array.shape[1]):  # For each feature
                feature_values = heatmap_array[:, col_idx]  # All clusters for this feature
                val = heatmap_array[row_idx, col_idx]
                
                # Min-max normalize within this feature
                feature_min = feature_values.min()
                feature_max = feature_values.max()
                if feature_max > feature_min:
                    normalized_val = (val - feature_min) / (feature_max - feature_min)
                else:
                    normalized_val = 0.5  # If all values are the same, use middle value
                
                normalized_row.append(normalized_val)
            normalized_by_row.append(normalized_row)
        
        normalized_array = np.array(normalized_by_row)
        
        # Create text matrices for both raw and normalized values
        text_matrix_raw = []
        text_matrix_norm = []
        for i in range(len(cluster_labels)):
            row_text_raw = []
            row_text_norm = []
            for j in range(len(feature_labels)):
                raw_val = heatmap_data[i][j]
                norm_val = normalized_array[i][j]
                
                # Format raw values
                if "pct" in feature_cols[j][0] or "%" in feature_labels[j]:
                    row_text_raw.append(f"{raw_val:.1f}%")
                elif "age" in feature_cols[j][0].lower():
                    row_text_raw.append(f"{raw_val:.0f}")
                elif "bmi" in feature_cols[j][0].lower():
                    row_text_raw.append(f"{raw_val:.1f}")
                else:
                    row_text_raw.append(f"{raw_val:.1f}")
                
                # Format normalized values
                row_text_norm.append(f"{norm_val:.2f}")
            
            text_matrix_raw.append(row_text_raw)
            text_matrix_norm.append(row_text_norm)
        
        # Create heatmap: Row-normalized values with YlOrRd colorscale (yellow to red)
        fig_raw = go.Figure(data=go.Heatmap(
            z=normalized_array.T,  # Transpose: features as rows, clusters as columns
            x=cluster_labels,
            y=feature_labels,
            colorscale='YlOrRd',  # Yellow to Orange to Red
            colorbar=dict(
                title=dict(
                    text="Normalized Value<br>(0=min, 1=max per feature)",
                    font=dict(size=12, color=TEXT_PRIMARY),
                ),
                tickfont=dict(size=10, color=TEXT_PRIMARY),
            ),
            text=[[text_matrix_raw[i][j] for i in range(len(cluster_labels))] for j in range(len(feature_labels))],
            texttemplate="%{text}",
            textfont=dict(size=9, color="#111827", family="Arial Black"),
            hovertemplate="<b>%{y}</b><br>Cluster: %{x}<br>Normalized: %{z:.3f}<br>Raw Value: %{text}<extra></extra>",
        ))
        
        fig_raw.update_layout(
            title=dict(
                text=f"<b>Clinical Profile Heatmap — {cohort_display}</b><br><sub>Row-normalized values (0=min, 1=max per feature) across clusters (k={metrics['optimal_k']})</sub>",
                font=dict(size=16, color=TEXT_PRIMARY),
                x=0.5,
            ),
            xaxis=dict(
                title=dict(text="<b>Cluster</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                tickfont=dict(size=11, color=TEXT_PRIMARY),
            ),
            yaxis=dict(
                title=dict(text="<b>Clinical Feature</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                tickfont=dict(size=11, color=TEXT_PRIMARY),
            ),
            plot_bgcolor=BG_CONTENT,
            paper_bgcolor=BG_CONTENT,
            font=dict(color=TEXT_PRIMARY),
            height=600,
            margin=dict(l=120, r=50, t=100, b=50),
        )
        
        st.plotly_chart(fig_raw, use_container_width=True, config={"displayModeBar": True})
    else:
        st.warning("No clinical features available for heatmap visualization.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Cluster comparison --------
    st.subheader("📊 Cluster Comparison Dashboard")

    comparison_data = create_cluster_comparison_data(cohort)

    if not comparison_data.empty:
        st.markdown("### Summary table")
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

        st.markdown("### Feature comparison across clusters")

        feature_cols = [
            "sbp_latest_median",
            "age_median",
            "bmi_latest_median",
            "encounter_count_12m_median",
            "icd3_count_median",
        ]
        available_features = [
            col for col in feature_cols if col in comparison_data.columns
        ]

        if available_features:
            # Create a better feature selection with display names
            feature_display_map = {
                "sbp_latest_median": "SBP (mmHg)",
                "age_median": "Age (years)",
                "bmi_latest_median": "BMI (kg/m²)",
                "encounter_count_12m_median": "Encounters/Year",
                "icd3_count_median": "ICD-3 Count",
            }
            
            feature_options = {
                feature_display_map.get(f, f.replace("_", " ").title()): f
                for f in available_features
            }
            
            selected_feature_display = st.selectbox(
                "Select feature to compare",
                options=list(feature_options.keys()),
                key=f"feature_comparison_{cohort}",
            )
            selected_feature = feature_options[selected_feature_display]

            # Prepare data
            cluster_labels = [
                f"Cluster {int(i)}" for i in comparison_data["cluster"].values
            ]
            values = comparison_data[selected_feature].values

            colors_professional = [
                "#38bdf8",
                "#22c55e",
                "#f97316",
                "#e879f9",
                "#a855f7",
                "#06b6d4",
            ]
            bar_colors = [
                colors_professional[i % len(colors_professional)]
                for i in range(len(cluster_labels))
            ]

            # Create interactive Plotly bar chart
            fig_comparison = go.Figure(
                data=[
                    go.Bar(
                        x=cluster_labels,
                        y=values,
                        marker=dict(
                color=bar_colors,
                            line=dict(color="#111827", width=1.5),
                        ),
                        text=[f"{val:.1f}" for val in values],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>%{yaxis.title.text}: %{y:.2f}<extra></extra>",
                    )
                ]
            )

            fig_comparison.update_layout(
                title=dict(
                    text=f"<b>{selected_feature_display} Across Clusters</b>",
                    font=dict(size=16, color=TEXT_PRIMARY),
                    x=0.5,
                ),
                xaxis=dict(
                    title=dict(text="<b>Cluster</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                    tickfont=dict(size=11, color=TEXT_PRIMARY),
                    gridcolor=BORDER_COLOR,
                    gridwidth=1,
                ),
                yaxis=dict(
                    title=dict(
                        text=f"<b>{selected_feature_display}</b>",
                        font=dict(size=12, color=TEXT_PRIMARY),
                    ),
                    tickfont=dict(size=11, color=TEXT_PRIMARY),
                    gridcolor=BORDER_COLOR,
                    gridwidth=1,
                ),
                plot_bgcolor=BG_CONTENT,
                paper_bgcolor=BG_CONTENT,
                font=dict(color=TEXT_PRIMARY),
                hovermode="closest",
                height=500,
                margin=dict(l=50, r=50, t=80, b=50),
            )

            st.plotly_chart(
                fig_comparison, use_container_width=True, config={"displayModeBar": True}
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Patient-level insights --------
    st.subheader("🔎 Patient-Level Insights")

    # Build patient list directly from this cohort's assignments
    if "assignments" in data and not data["assignments"].empty:
        assignments = data["assignments"].copy()

        # Note: assignments are already cohort-specific (loaded from cohort-specific files)
        # Only filter by cohort column if it exists AND the values don't match the current cohort
        # This handles edge cases where a combined assignments file might exist
        if "cohort" in assignments.columns:
            # Normalize cohort values for comparison (handle case differences)
            assignments_cohort_normalized = assignments["cohort"].astype(str).str.lower().str.strip()
            cohort_normalized = str(cohort).lower().strip()
            assignments = assignments[assignments_cohort_normalized == cohort_normalized]

        # Now build patient list from this cohort's assignments only
        patient_list = sorted(assignments["pid"].unique().tolist())

        if len(patient_list) == 0:
            st.warning(
                f"No patients found in assignments for cohort {cohort_display}. "
                "Please check the segmentation data."
            )
        else:
            st.markdown(
                f"<div style='font-size:0.85rem; color:{TEXT_MUTED}; margin-bottom:0.5rem;'>"
                f"Available patients in {cohort_display}: <strong>{len(patient_list):,}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

            selectbox_key = f"patient_selectbox_{cohort}"

            # Determine initial index based on current selection (per cohort)
            # Reset selection if it's not in the current patient list
            initial_index = 0
            if selectbox_key in st.session_state:
                current_selection = st.session_state[selectbox_key]
                if current_selection and current_selection in patient_list:
                    try:
                        idx = patient_list.index(current_selection)
                        initial_index = idx + 1  # +1 because first option is empty string
                        # Ensure index is within bounds
                        if initial_index >= len(patient_list) + 1:
                            initial_index = 0
                    except (ValueError, AttributeError, IndexError):
                        # Patient not in list or invalid, reset to empty selection
                        st.session_state[selectbox_key] = ""
                        initial_index = 0
                else:
                    # Selection is empty or not in current list, reset
                    st.session_state[selectbox_key] = ""
                    initial_index = 0
            else:
                # First time for this cohort, initialize with empty selection
                st.session_state[selectbox_key] = ""
                initial_index = 0

            # Ensure initial_index is always valid (0 to len(patient_list))
            max_index = len(patient_list)  # max index is len(patient_list) because we add "" at the start
            if initial_index < 0 or initial_index > max_index:
                initial_index = 0

            selected_patient = st.selectbox(
                "Patient identifier",
                options=[""] + patient_list,
                format_func=lambda x: "Select a patient..."
                if x == ""
                else f"{x[:20]}..." if len(x) > 20 else x,
                help=f"Select a patient from the {cohort_display} cohort ({len(patient_list):,} patients available)",
                key=selectbox_key,
                index=initial_index,
            )

            if selected_patient:
                # Ensure the patient truly belongs to this cohort's assignments
                if selected_patient not in patient_list:
                    st.warning(
                        f"⚠️ Patient **{selected_patient}** is not in the {cohort_display} cohort. "
                        f"Please select a patient from the {cohort_display} patient list."
                    )
                else:
                    # Get cluster_id directly from assignments to avoid any cohort mismatch
                    row = assignments[assignments["pid"] == selected_patient]
                    if row.empty:
                        st.warning(
                            f"Patient {selected_patient} is not present in the {cohort_display} cohort."
                        )
                    else:
                        cluster_id = int(row["cluster"].iloc[0])

                        # Comparison with cluster
                        # Pass cluster_id directly since we already have it from assignments
                        # This avoids the issue where get_patient_cluster might not find the patient
                        # in the cohort's assignments file (e.g., if patient exists in patient_summary
                        # but not in the specific cohort's assignments file)
                        comparison = compare_patient_to_cluster(selected_patient, cohort, cluster_id=cluster_id)

                        # Debug output to diagnose z01 patient data retrieval issues
                        if not comparison:
                            with st.expander("🔍 Debug Information (click to expand)", expanded=False):
                                st.write("**Diagnostic Information:**")
                                
                                # Check each step
                                cluster_id_check = get_patient_cluster(selected_patient, cohort)
                                patient_details_check = get_patient_details(selected_patient, cohort)
                                cluster_summary_check = get_cluster_summary(cohort, cluster_id)
                                
                                st.write(f"- **Patient ID:** {selected_patient}")
                                st.write(f"- **Cohort:** {cohort}")
                                st.write(f"- **Cluster ID from lookup:** {cluster_id_check}")
                                st.write(f"- **Cluster ID from assignments:** {cluster_id}")
                                st.write(f"- **Patient details found:** {patient_details_check is not None}")
                                st.write(f"- **Cluster summary found:** {cluster_summary_check is not None and len(cluster_summary_check) > 0}")
                                
                                if patient_details_check:
                                    st.write(f"- **Patient details keys:** {list(patient_details_check.keys())[:15]}")
                                    # Show key metrics
                                    key_metrics = ['sbp_latest', 'age', 'encounter_count_12m', 'icd3_count', 'bmi_latest']
                                    for metric in key_metrics:
                                        if metric in patient_details_check:
                                            st.write(f"  - {metric}: {patient_details_check[metric]}")
                                
                                if cluster_summary_check:
                                    st.write(f"- **Cluster summary keys:** {list(cluster_summary_check.keys())[:15]}")
                                    # Show key medians
                                    key_medians = ['sbp_latest_median', 'age_median', 'encounter_count_12m_median', 'icd3_count_median', 'bmi_latest_median']
                                    for median in key_medians:
                                        if median in cluster_summary_check:
                                            st.write(f"  - {median}: {cluster_summary_check[median]}")

                        if comparison:
                            st.success(
                                f"Patient **{selected_patient}** is assigned to **Cluster {cluster_id}**."
                            )

                            cluster_info = get_cluster_summary(cohort, cluster_id)

                            col_info1, col_info2 = st.columns(2)

                            with col_info1:
                                st.markdown(
                                    f"""
                                    <div class="info-box">
                                        <h4>Cluster summary</h4>
                                        <p><strong>Cluster label:</strong> {cluster_info.get('cluster_name', 'N/A')}</p>
                                        <p><strong>Cluster size:</strong> {int(cluster_info.get('n_patients', 0)):,} patients ({cluster_info.get('pct_patients', 0):.1f}%)</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with col_info2:
                                st.markdown(
                                    f"""
                                    <div class="success-box">
                                        <h4>Patient assignment</h4>
                                        <p><strong>Patient ID:</strong> {selected_patient}</p>
                                        <p><strong>Assigned cluster:</strong> Cluster {cluster_id}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            st.markdown("<hr/>", unsafe_allow_html=True)

                            st.markdown("### Patient characteristics vs cluster median")

                            if comparison["patient"]:
                                comparison_data_rows = []
                                for metric_name in comparison["patient"].keys():
                                    patient_val = comparison["patient"][metric_name]
                                    cluster_val = comparison["cluster"][metric_name]
                                    difference = comparison["difference"][metric_name]

                                    comparison_data_rows.append(
                                        {
                                            "Metric": metric_name,
                                            "Patient value": patient_val,
                                            "Cluster median": cluster_val,
                                            "Difference": difference,
                                            "Status": "Above"
                                            if difference > 0
                                            else "Below"
                                            if difference < 0
                                            else "Equal",
                                        }
                                    )

                                comparison_df = pd.DataFrame(comparison_data_rows)

                                st.dataframe(
                                    comparison_df.style.format(
                                        {
                                            "Patient value": "{:.2f}",
                                            "Cluster median": "{:.2f}",
                                            "Difference": "{:.2f}",
                                        }
                                    ).background_gradient(
                                        subset=["Difference"],
                                        cmap="RdYlGn",
                                        vmin=-50,
                                        vmax=50,
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                st.markdown("### PCA Visualization: Patient Position in Cluster Space")
                                
                                # Get PCA coordinates for visualization
                                try:
                                    # Get all assignments for this cohort
                                    all_assignments = data["assignments"].copy()
                                    
                                    # Get medoids
                                    medoids_df = data["medoids"].copy()
                                    
                                    # Prepare features for PCA (use same features as clustering)
                                    feature_cols_for_pca = []
                                    if "sbp_latest" in all_assignments.columns:
                                        feature_cols_for_pca.append("sbp_latest")
                                    if "age" in all_assignments.columns:
                                        feature_cols_for_pca.append("age")
                                    if "encounter_count_12m" in all_assignments.columns:
                                        feature_cols_for_pca.append("encounter_count_12m")
                                    if "icd3_count" in all_assignments.columns:
                                        feature_cols_for_pca.append("icd3_count")
                                    if "bmi_latest" in all_assignments.columns:
                                        feature_cols_for_pca.append("bmi_latest")
                                    
                                    # Add binary features if available
                                    binary_features = [col for col in all_assignments.columns 
                                                     if col.startswith("has_") and col != "has_I10" or col == "has_I10"]
                                    feature_cols_for_pca.extend([f for f in binary_features if f in all_assignments.columns][:4])
                                    
                                    # Filter to available features
                                    available_pca_features = [f for f in feature_cols_for_pca if f in all_assignments.columns]
                                    
                                    if len(available_pca_features) >= 2:
                                        # Prepare data for PCA
                                        pca_data = all_assignments[["pid", "cluster"] + available_pca_features].copy()
                                        
                                        # Handle missing values - fill with median for numeric, mode for binary
                                        for col in available_pca_features:
                                            if pca_data[col].dtype in ['int64', 'float64']:
                                                pca_data[col] = pca_data[col].fillna(pca_data[col].median())
                                            else:
                                                pca_data[col] = pca_data[col].fillna(pca_data[col].mode()[0] if len(pca_data[col].mode()) > 0 else 0)
                                        
                                        # Standardize features
                                        from sklearn.preprocessing import StandardScaler
                                        from sklearn.decomposition import PCA
                                        
                                        scaler = StandardScaler()
                                        X_scaled = scaler.fit_transform(pca_data[available_pca_features])
                                        
                                        # Compute PCA
                                        pca = PCA(n_components=2)
                                        pca_coords = pca.fit_transform(X_scaled)
                                        
                                        # Add PCA coordinates to dataframe
                                        pca_data["PC1"] = pca_coords[:, 0]
                                        pca_data["PC2"] = pca_coords[:, 1]
                                        
                                        # Get selected patient's coordinates
                                        patient_pca = pca_data[pca_data["pid"] == selected_patient]
                                        
                                        if not patient_pca.empty:
                                            patient_pc1 = patient_pca["PC1"].iloc[0]
                                            patient_pc2 = patient_pca["PC2"].iloc[0]
                                            patient_cluster_id = int(patient_pca["cluster"].iloc[0])
                                            
                                            # Get medoid coordinates
                                            medoid_coords = []
                                            for _, medoid_row in medoids_df.iterrows():
                                                medoid_pid = medoid_row["medoid_pid"]
                                                medoid_cluster = int(medoid_row["cluster"])
                                                medoid_data = pca_data[pca_data["pid"] == medoid_pid]
                                                if not medoid_data.empty:
                                                    medoid_coords.append({
                                                        "cluster": medoid_cluster,
                                                        "pid": medoid_pid,
                                                        "PC1": medoid_data["PC1"].iloc[0],
                                                        "PC2": medoid_data["PC2"].iloc[0],
                                                    })
                                            
                                            # Create interactive scatter plot
                                            fig_pca = go.Figure()
                                            
                                            # Color scheme matching the image
                                            cluster_colors = {
                                                0: "#ef4444",  # Red
                                                1: "#06b6d4",  # Teal/Cyan
                                                2: "#eab308",  # Yellow
                                                3: "#10b981",  # Green
                                                4: "#a855f7",  # Purple
                                                5: "#f97316",  # Orange
                                            }
                                            
                                            # Plot each cluster
                                            for cluster_id in sorted(pca_data["cluster"].unique()):
                                                cluster_data = pca_data[pca_data["cluster"] == cluster_id]
                                                cluster_name = profiles[profiles["cluster"] == cluster_id]["cluster_name"].iloc[0] if "cluster_name" in profiles.columns else f"Cluster {int(cluster_id)}"
                                                
                                                # Get cluster size
                                                cluster_size = len(cluster_data)
                                                
                                                fig_pca.add_trace(
                                                    go.Scatter(
                                                        x=cluster_data["PC1"],
                                                        y=cluster_data["PC2"],
                                                        mode="markers",
                                                        name=f"C{int(cluster_id)}: {cluster_name} (n={cluster_size})",
                                                        marker=dict(
                                                            color=cluster_colors.get(int(cluster_id), "#6b7280"),
                                                            size=6,
                                                            opacity=0.6,
                                                            line=dict(width=0.5, color="#111827"),
                                                        ),
                                                        hovertemplate="<b>%{fullData.name}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                                                    )
                                                )
                                            
                                            # Plot medoids as stars
                                            for medoid in medoid_coords:
                                                medoid_cluster = medoid["cluster"]
                                                medoid_name = profiles[profiles["cluster"] == medoid_cluster]["cluster_name"].iloc[0] if "cluster_name" in profiles.columns else f"Cluster {medoid_cluster}"
                                                
                                                fig_pca.add_trace(
                                                    go.Scatter(
                                                        x=[medoid["PC1"]],
                                                        y=[medoid["PC2"]],
                                                        mode="markers",
                                                        name=f"Medoid C{medoid_cluster}",
                                                        marker=dict(
                                                            symbol="star",
                                                            size=20,
                                                            color=cluster_colors.get(medoid_cluster, "#6b7280"),
                                                            line=dict(width=2, color="#111827"),
                                                        ),
                                                        hovertemplate=f"<b>Medoid: {medoid_name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
                                                        showlegend=True,
                                                    )
                                                )
                                            
                                            # Highlight selected patient (use cluster color)
                                            patient_cluster_color = cluster_colors.get(patient_cluster_id, "#6b7280")
                                            fig_pca.add_trace(
                                                go.Scatter(
                                                    x=[patient_pc1],
                                                    y=[patient_pc2],
                                                    mode="markers",
                                                    name=f"Selected Patient: {selected_patient[:20]}...",
                                                    marker=dict(
                                                        symbol="diamond",
                                                        size=25,
                                                        color=patient_cluster_color,  # Match cluster color
                                                        line=dict(width=3, color="#111827"),
                                                    ),
                                                    hovertemplate=f"<b>Selected Patient</b><br>Cluster: {patient_cluster_id}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
                                                    showlegend=True,
                                                )
                                            )
                                            
                                            # Calculate explained variance
                                            explained_var = pca.explained_variance_ratio_
                                            
                                            fig_pca.update_layout(
                                                title=dict(
                                                    text=f"<b>Patient Clusters in Principal Component Space — {cohort_display}</b><br><sub>PC1: {explained_var[0]*100:.1f}% variance | PC2: {explained_var[1]*100:.1f}% variance</sub>",
                                                    font=dict(size=16, color=TEXT_PRIMARY),
                                                    x=0.5,
                                                ),
                                                xaxis=dict(
                                                    title=dict(text="<b>PC1 (First Principal Component)</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                                                    tickfont=dict(size=11, color=TEXT_PRIMARY),
                                                    gridcolor=BORDER_COLOR,
                                                    gridwidth=1,
                                                ),
                                                yaxis=dict(
                                                    title=dict(text="<b>PC2 (Second Principal Component)</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                                                    tickfont=dict(size=11, color=TEXT_PRIMARY),
                                                    gridcolor=BORDER_COLOR,
                                                    gridwidth=1,
                                                ),
                                                plot_bgcolor=BG_CONTENT,
                                                paper_bgcolor=BG_CONTENT,
                                                font=dict(color=TEXT_PRIMARY),
                                                hovermode="closest",
                                                legend=dict(
                                                    font=dict(size=10, color=TEXT_PRIMARY),
                                                    bgcolor="rgba(0,0,0,0)",
                                                    bordercolor=BORDER_COLOR,
                                                    borderwidth=1,
                                                    yanchor="top",
                                                    y=0.99,
                                                    xanchor="right",
                                                    x=0.99,
                                                ),
                                                height=600,
                                                margin=dict(l=50, r=50, t=100, b=50),
                                            )
                                            
                                            st.plotly_chart(
                                                fig_pca, use_container_width=True, config={"displayModeBar": True}
                                            )
                                        else:
                                            st.warning("Could not find PCA coordinates for selected patient.")
                                    else:
                                        st.warning("Insufficient features available for PCA visualization.")
                                except Exception as e:
                                    st.warning(f"Could not generate PCA visualization: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())

                                st.markdown("### Visual comparison")

                                metrics_list = list(comparison["patient"].keys())
                                patient_values = [
                                    comparison["patient"][m] for m in metrics_list
                                ]
                                cluster_values = [
                                    comparison["cluster"][m] for m in metrics_list
                                ]

                                # Create interactive grouped bar chart with Plotly
                                fig_comparison = go.Figure()

                                # Patient bars
                                fig_comparison.add_trace(
                                    go.Bar(
                                        name="Patient",
                                        x=metrics_list,
                                        y=patient_values,
                                        marker=dict(
                                    color="#38bdf8",
                                            line=dict(color="#111827", width=1.5),
                                        ),
                                        text=[f"{val:.1f}" for val in patient_values],
                                        textposition="outside",
                                        hovertemplate="<b>%{x}</b><br>Patient: %{y:.2f}<extra></extra>",
                                )
                                )

                                # Cluster median bars
                                fig_comparison.add_trace(
                                    go.Bar(
                                        name="Cluster Median",
                                        x=metrics_list,
                                        y=cluster_values,
                                        marker=dict(
                                    color="#22c55e",
                                            line=dict(color="#111827", width=1.5),
                                        ),
                                        text=[f"{val:.1f}" for val in cluster_values],
                                        textposition="outside",
                                        hovertemplate="<b>%{x}</b><br>Cluster Median: %{y:.2f}<extra></extra>",
                                    )
                                )

                                fig_comparison.update_layout(
                                    title=dict(
                                        text=f"<b>Patient vs Cluster Comparison — Cluster {cluster_id}</b>",
                                        font=dict(size=16, color=TEXT_PRIMARY),
                                        x=0.5,
                                    ),
                                    xaxis=dict(
                                        title=dict(text="<b>Metric</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                                        tickfont=dict(size=11, color=TEXT_PRIMARY),
                                        tickangle=-45,
                                        gridcolor=BORDER_COLOR,
                                        gridwidth=1,
                                    ),
                                    yaxis=dict(
                                        title=dict(text="<b>Value</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                                        tickfont=dict(size=11, color=TEXT_PRIMARY),
                                        gridcolor=BORDER_COLOR,
                                        gridwidth=1,
                                    ),
                                    barmode="group",
                                    plot_bgcolor=BG_CONTENT,
                                    paper_bgcolor=BG_CONTENT,
                                    font=dict(color=TEXT_PRIMARY),
                                    hovermode="closest",
                                    legend=dict(
                                        font=dict(size=11, color=TEXT_PRIMARY),
                                        bgcolor="rgba(0,0,0,0)",
                                        bordercolor=BORDER_COLOR,
                                        borderwidth=1,
                                    ),
                                    height=500,
                                    margin=dict(l=50, r=50, t=80, b=150),
                                )

                                st.plotly_chart(
                                    fig_comparison, use_container_width=True, config={"displayModeBar": True}
                                )

                                st.markdown("### Summary insights")

                                above_avg = [
                                    m
                                    for m, diff in comparison["difference"].items()
                                    if diff > 0
                                ]
                                below_avg = [
                                    m
                                    for m, diff in comparison["difference"].items()
                                    if diff < 0
                                ]

                                col_insight1, col_insight2 = st.columns(2)

                                with col_insight1:
                                    if above_avg:
                                        st.markdown(
                                            f"""
                                            <div class="success-box">
                                                <h4>Above cluster median</h4>
                                                <ul>
                                                    {''.join([f'<li>{metric}</li>' for metric in above_avg])}
                                                </ul>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                with col_insight2:
                                    if below_avg:
                                        st.markdown(
                                            f"""
                                            <div class="warning-box">
                                                <h4>Below cluster median</h4>
                                                <ul>
                                                    {''.join([f'<li>{metric}</li>' for metric in below_avg])}
                                                </ul>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                            else:
                                st.info(
                                    "Patient data is available but detailed comparison metrics are not present."
                                )
                        else:
                            st.warning(
                                f"Comparison data for patient {selected_patient} could not be loaded."
                            )
    else:
        st.warning(
            "Patient list not available for this cohort. Please verify that the assignments table is present."
        )

# -----------------------------------------------------------------------------
# DRUG ADHERENCE FORECASTING MODULE
# -----------------------------------------------------------------------------
elif st.session_state["section"] == "adherence":
    st.header("💊 Drug Adherence Prediction Workspace")

    time_period = time_period or "Monthly"
    gran_label = (
        "Monthly refill adherence"
        if time_period == "Monthly"
        else "Bi-weekly refill adherence"
    )
    st.subheader(f"📅 Aggregation: {gran_label}")

    st.markdown(
        f"""
        <div class="info-box">
            <h3 style="margin-bottom:4px;">Adherence forecasting scope</h3>
            <p style="margin-bottom:0;">Adherence is modelled as a time series using <strong>{time_period.lower()}</strong> level aggregation, with XGBoost as the primary forecasting model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Default forecast configuration values
    model_choice = "XGBoost"
    forecast_periods = 6
    train_size = 83
    show_confidence = True
    show_technical = False

    # Load adherence data
    @st.cache_data
    def load_adherence(period="monthly"):
        try:
            if period == "bi-weekly":
                data_path = Path("time-series-drug-adherence/data/biweekly_overall.csv")
                if not data_path.exists():
                    st.error(f"Bi-weekly data file not found: {data_path}")
                    return None
                df = pd.read_csv(data_path)
                df["biweek_start"] = pd.to_datetime(df["biweek_start"])
                df_valid = df[df["refill_adherence"].notna()].copy()
                df_valid = df_valid.rename(columns={"biweek_start": "month_start"})
                return df_valid
            else:
                return load_adherence_data()
        except Exception as e:
            st.error(f"Error loading adherence data: {e}")
            return None

    adherence_data = load_adherence(
        "bi-weekly" if time_period == "Bi-weekly" else "monthly"
    )

    if adherence_data is None:
        st.error("Failed to load adherence data. Please check data files.")
        st.stop()

    period_label = "bi-weeks" if time_period == "Bi-weekly" else "months"

    data_ts = adherence_data.set_index("month_start").sort_index()
    target_col = "refill_adherence"
    n_train = int(len(data_ts) * train_size / 100)
    train_data = data_ts.iloc[:n_train]
    test_data = data_ts.iloc[n_train:]

    # -------- Adherence overview --------
    st.subheader("📊 Adherence Overview")

    # XGBoost Model Performance Metrics - Load from pre-trained results
    xgb_metrics = None
    try:
        # Load metrics from pre-trained model results
        if time_period == "Bi-weekly":
            metrics_path = Path("time-series-drug-adherence/data/model_performance_metrics_biweekly.csv")
        else:
            metrics_path = Path("time-series-drug-adherence/data/model_performance_metrics.csv")
        
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            xgb_row = metrics_df[metrics_df["Model"] == "XGBoost"]
            
            if not xgb_row.empty:
                xgb_metrics = {
                    "MAE": float(xgb_row["MAE"].iloc[0]),
                    "RMSE": float(xgb_row["RMSE"].iloc[0]),
                    "MAPE": float(xgb_row["MAPE (%)"].iloc[0])
                }
    except Exception as e:
        pass  # Silently fail if we can't load metrics
    
    if xgb_metrics is not None:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("**XGBoost Model Performance**")

        col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
                "MAE",
                f"{xgb_metrics['MAE']:.2f}",
                help="Mean Absolute Error - average prediction error in percentage points (lower is better)",
        )

    with col2:
        st.metric(
                "RMSE",
                f"{xgb_metrics['RMSE']:.2f}",
                help="Root Mean Squared Error - penalizes larger errors more (lower is better)",
        )

    with col3:
        st.metric(
                "MAPE",
                f"{xgb_metrics['MAPE']:.2f}%",
                help="Mean Absolute Percentage Error - average percentage error (lower is better)",
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Model information
    st.markdown(
        """
        <div class="info-box">
            <h4 style="margin-bottom:4px;">Forecasting Model</h4>
            <p style="margin-bottom:0;">XGBoost is the primary forecasting engine for adherence prediction, validated on historical time series data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------- Trend visualization --------
    st.markdown("### Adherence trend over time")

    # Create interactive Plotly chart matching the workspace style
    fig_trend = go.Figure()
    
    # Separate train and test data (matching workspace style)
    train_data_plot = data_ts.iloc[:n_train].copy()
    test_data_plot = data_ts.iloc[n_train:].copy() if n_train < len(data_ts) else pd.DataFrame()
    
    # Check if target column exists and has data
    if target_col not in data_ts.columns:
        st.error(f"Column '{target_col}' not found in data. Available columns: {list(data_ts.columns)}")
        st.stop()
    
    # Determine split date
    split_date = train_data_plot.index[-1] if len(train_data_plot) > 0 else data_ts.index[0]
    if len(test_data_plot) > 0:
        split_date = test_data_plot.index[0]
    
    # Add training period shaded area (light blue/teal, matching image)
    if len(train_data_plot) > 0:
        train_start = train_data_plot.index[0]
        train_end = train_data_plot.index[-1]
        fig_trend.add_vrect(
            x0=train_start,
            x1=train_end,
            fillcolor="rgba(56, 189, 248, 0.15)",  # Light blue/teal
            layer="below",
            line_width=0,
            annotation_text="Training Period",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#38bdf8",
        )
    
    # Add test period shaded area (light orange, matching image)
    if len(test_data_plot) > 0:
        test_start = test_data_plot.index[0]
        test_end = test_data_plot.index[-1]
        fig_trend.add_vrect(
            x0=test_start,
            x1=test_end,
            fillcolor="rgba(249, 115, 22, 0.15)",  # Light orange
            layer="below",
            line_width=0,
            annotation_text="Test Period",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="#f97316",
        )
    
    # Convert adherence to percentage if needed (handle both 0-1 and 0-100 formats)
    def convert_to_percentage(series):
        """Convert adherence values to percentage if needed"""
        if series.max() <= 1 and series.min() >= 0:
            return series * 100
        return series
    
    # Plot training data (blue with circular markers, matching image)
    if len(train_data_plot) > 0:
        train_y = convert_to_percentage(train_data_plot[target_col].dropna())
        if len(train_y) > 0:
            fig_trend.add_trace(
                go.Scatter(
                    x=train_data_plot.index[train_data_plot[target_col].notna()],
                    y=train_y,
                    mode="lines+markers",
                    name="Training Data",
                    line=dict(color="#38bdf8", width=3),
                    marker=dict(size=8, color="#38bdf8", symbol="circle"),
                    hovertemplate="<b>%{x}</b><br>Training: %{y:.2f}%<extra></extra>",
                )
            )
    
    # Plot test data (orange with square markers, matching image)
    if len(test_data_plot) > 0:
        test_y = convert_to_percentage(test_data_plot[target_col].dropna())
        if len(test_y) > 0:
            fig_trend.add_trace(
                go.Scatter(
                    x=test_data_plot.index[test_data_plot[target_col].notna()],
                    y=test_y,
                    mode="lines+markers",
                    name="Test Data",
                    line=dict(color="#f97316", width=3),
                    marker=dict(size=8, color="#f97316", symbol="square"),
                    hovertemplate="<b>%{x}</b><br>Test: %{y:.2f}%<extra></extra>",
                )
            )
    
    # Add train/test split line (red dashed, matching image)
    if len(test_data_plot) > 0:
        split_date_display = split_date.strftime('%Y-%m') if hasattr(split_date, 'strftime') else str(split_date)[:7]
        fig_trend.add_shape(
            type="line",
            x0=split_date,
            x1=split_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#ef4444", width=2, dash="dash"),
        )
        fig_trend.add_annotation(
            x=split_date,
            y=1,
            yref="paper",
            text=f"Train/Test Split ({split_date_display})",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=11, color="#ef4444"),
        )
    
    # Add threshold lines (matching image)
    fig_trend.add_hline(
        y=80,
        line_dash="dot",
        line_color=ACCENT_DANGER,
        line_width=2,
        opacity=0.6,
        annotation_text="Minimum threshold (80%)",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color=ACCENT_DANGER,
    )
    
    fig_trend.add_hline(
        y=95,
        line_dash="dot",
        line_color="#22c55e",
        line_width=2,
        opacity=0.6,
        annotation_text="Excellence threshold (95%)",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="#22c55e",
    )
    
    # Add target zone fill (between 80% and 95%, matching image)
    fig_trend.add_hrect(
        y0=80,
        y1=95,
        fillcolor="rgba(34, 197, 94, 0.08)",
        layer="below",
        line_width=0,
    )
    
    # Update layout (matching image style)
    x_label = "Bi-week" if time_period == "Bi-weekly" else "Month"
    
    # Determine y-axis range based on data
    all_values = []
    if len(train_data_plot) > 0:
        train_vals = convert_to_percentage(train_data_plot[target_col].dropna())
        all_values.extend(train_vals.tolist())
    if len(test_data_plot) > 0:
        test_vals = convert_to_percentage(test_data_plot[target_col].dropna())
        all_values.extend(test_vals.tolist())
    
    if all_values:
        y_min = max(70, min(all_values) * 0.95)
        y_max = min(100, max(all_values) * 1.05)
    else:
        y_min = 70
        y_max = 100
    
    fig_trend.update_layout(
        title=dict(
            text=f"<b>Adherence Rate — {time_period}</b>",
            font=dict(size=16, color=TEXT_PRIMARY),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text=f"<b>{x_label}</b>", font=dict(size=13, color=TEXT_PRIMARY)),
            tickfont=dict(size=11, color=TEXT_PRIMARY),
            gridcolor=BORDER_COLOR,
            gridwidth=1,
            tickangle=-45,
        ),
        yaxis=dict(
            title=dict(text="<b>Refill adherence (%)</b>", font=dict(size=13, color=TEXT_PRIMARY)),
            tickfont=dict(size=11, color=TEXT_PRIMARY),
            gridcolor=BORDER_COLOR,
            gridwidth=1,
            range=[y_min, y_max],
        ),
        plot_bgcolor=BG_CONTENT,
        paper_bgcolor=BG_CONTENT,
        font=dict(color=TEXT_PRIMARY),
        hovermode="x unified",
        legend=dict(
            font=dict(size=11, color=TEXT_PRIMARY),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=100),
    )
    
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": True})

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Model performance --------
    st.subheader("🤖 Forecast Model Performance")
    
    # Load predictions vs actuals from pre-trained results
    st.markdown("### Predictions vs Actuals")
    
    try:
        # Load predictions data
        if time_period == "Bi-weekly":
            predictions_path = Path("time-series-drug-adherence/data/test_predictions_all_models_biweekly.csv")
            model_col = "Prophet"  # Use Prophet for bi-weekly
        else:
            predictions_path = Path("time-series-drug-adherence/data/test_predictions_all_models.csv")
            model_col = "XGBoost"  # Use XGBoost for monthly
        
        if predictions_path.exists():
            pred_df = pd.read_csv(predictions_path)
            
            # Determine date column name
            date_col = "Month" if "Month" in pred_df.columns else "Date"
            
            # Convert date column to datetime
            pred_df[date_col] = pd.to_datetime(pred_df[date_col])
            
            # Get actual and predicted values
            if "Actual" in pred_df.columns and model_col in pred_df.columns:
                # Convert to percentage if needed (values are in 0-1 format)
                actual_values = pred_df["Actual"] * 100 if pred_df["Actual"].max() <= 1 else pred_df["Actual"]
                predicted_values = pred_df[model_col] * 100 if pred_df[model_col].max() <= 1 else pred_df[model_col]
                
                # Create interactive Plotly chart
                fig_predictions = go.Figure()
                
                # Add actual values (orange line with circles, more visible)
                fig_predictions.add_trace(
                    go.Scatter(
                        x=pred_df[date_col],
                        y=actual_values,
                        mode="lines+markers",
                        name="Actual",
                        line=dict(color="#f97316", width=3),
                        marker=dict(size=8, color="#f97316", symbol="circle"),
                        hovertemplate="<b>%{x}</b><br>Actual: %{y:.2f}%<extra></extra>",
                    )
                )
                
                # Add predicted values (blue line with squares, matching workspace style)
                fig_predictions.add_trace(
                    go.Scatter(
                        x=pred_df[date_col],
                        y=predicted_values,
                        mode="lines+markers",
                        name="Predicted",
                        line=dict(color="#38bdf8", width=2.5, dash="dash"),
                        marker=dict(size=7, color="#38bdf8", symbol="square"),
                        hovertemplate="<b>%{x}</b><br>Predicted: %{y:.2f}%<extra></extra>",
                    )
                )
                
                # Add threshold lines
                fig_predictions.add_hline(
                    y=80,
                    line_dash="dot",
                    line_color=ACCENT_DANGER,
                    line_width=2,
                    opacity=0.5,
                    annotation_text="Minimum threshold (80%)",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color=ACCENT_DANGER,
                )
                
                fig_predictions.add_hline(
                    y=95,
                    line_dash="dot",
                    line_color="#22c55e",
                    line_width=2,
                    opacity=0.5,
                    annotation_text="Excellence threshold (95%)",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="#22c55e",
                )
                
                # Determine y-axis range
                all_vals = list(actual_values) + list(predicted_values)
                y_min = max(70, min(all_vals) * 0.95) if all_vals else 70
                y_max = min(100, max(all_vals) * 1.05) if all_vals else 100
                
                # Update layout
                x_label = "Bi-week" if time_period == "Bi-weekly" else "Month"
                model_name = "Prediction Model"
                
                fig_predictions.update_layout(
                    title=dict(
                        text=f"<b>{model_name} Predictions vs Actual Adherence — {time_period}</b>",
                        font=dict(size=16, color=TEXT_PRIMARY),
                        x=0.5,
                    ),
                    xaxis=dict(
                        title=dict(text=f"<b>{x_label}</b>", font=dict(size=13, color=TEXT_PRIMARY)),
                        tickfont=dict(size=11, color=TEXT_PRIMARY),
                        gridcolor=BORDER_COLOR,
                        gridwidth=1,
                        tickangle=-45,
                    ),
                    yaxis=dict(
                        title=dict(text="<b>Refill adherence (%)</b>", font=dict(size=13, color=TEXT_PRIMARY)),
                        tickfont=dict(size=11, color=TEXT_PRIMARY),
                        gridcolor=BORDER_COLOR,
                        gridwidth=1,
                        range=[y_min, y_max],
                    ),
                    plot_bgcolor=BG_CONTENT,
                    paper_bgcolor=BG_CONTENT,
                    font=dict(color=TEXT_PRIMARY),
                    hovermode="x unified",
                    legend=dict(
                        font=dict(size=11, color=TEXT_PRIMARY),
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor=BORDER_COLOR,
                        borderwidth=1,
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                    ),
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=100),
                )
                
                st.plotly_chart(fig_predictions, use_container_width=True, config={"displayModeBar": True})
            else:
                st.warning(f"Required columns not found in predictions file. Expected 'Actual' and '{model_col}'.")
        else:
            st.warning(f"Predictions file not found: {predictions_path}")
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Residual Analysis --------
    st.subheader("📊 Residual Analysis")
    
    st.markdown("""
    <div class="info-box" style="margin-bottom:1rem;">
        <p style="margin-bottom:0; font-size:0.9rem;">
            Residual analysis evaluates model performance by examining the difference between actual and predicted values. 
            Ideal residuals should be randomly distributed around zero with no clear patterns.
        </p>
            </div>
    """, unsafe_allow_html=True)
    
    try:
        # Load predictions data (same as above)
        if time_period == "Bi-weekly":
            predictions_path = Path("time-series-drug-adherence/data/test_predictions_all_models_biweekly.csv")
            model_col = "Prophet"  # Use Prophet for bi-weekly
        else:
            predictions_path = Path("time-series-drug-adherence/data/test_predictions_all_models.csv")
            model_col = "XGBoost"  # Use XGBoost for monthly
        
        if predictions_path.exists():
            pred_df = pd.read_csv(predictions_path)
            
            # Determine date column name
            date_col = "Month" if "Month" in pred_df.columns else "Date"
            
            # Convert date column to datetime
            pred_df[date_col] = pd.to_datetime(pred_df[date_col])
            
            # Get actual and predicted values
            if "Actual" in pred_df.columns and model_col in pred_df.columns:
                # Convert to percentage if needed (values are in 0-1 format)
                actual_values = pred_df["Actual"] * 100 if pred_df["Actual"].max() <= 1 else pred_df["Actual"]
                predicted_values = pred_df[model_col] * 100 if pred_df[model_col].max() <= 1 else pred_df[model_col]
                
                # Calculate residuals (Actual - Predicted) in percentage points
                residuals = actual_values - predicted_values
                
                # Calculate statistics
                resid_mean = float(np.mean(residuals))
                resid_std = float(np.std(residuals))
                resid_min = float(np.min(residuals))
                resid_max = float(np.max(residuals))
                
                st.markdown("### Residuals Over Time")
                
                # Create residual time series plot
                fig_residuals = go.Figure()
                
                # Add residual line
                fig_residuals.add_trace(
                    go.Scatter(
                        x=pred_df[date_col],
                        y=residuals,
                        mode="lines+markers",
                        name="Residuals",
                        line=dict(color="#8b5cf6", width=3),
                        marker=dict(size=8, color="#8b5cf6"),
                        fill="tozeroy",
                        fillcolor="rgba(139, 92, 246, 0.2)",
                        hovertemplate="<b>%{x}</b><br>Residual: %{y:.2f}%pts<extra></extra>",
                    )
                )
                
                # Add zero reference line
                fig_residuals.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="#6b7280",
                    line_width=2,
                    opacity=0.7,
                    annotation_text="Zero (Perfect Prediction)",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="#6b7280",
                )
                
                # Add ±1 std dev bands
                fig_residuals.add_hrect(
                    y0=-resid_std,
                    y1=resid_std,
                    fillcolor="rgba(34, 197, 94, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text=f"±1 Std Dev ({resid_std:.2f}%pts)",
                    annotation_position="right",
                    annotation_font_size=9,
                    annotation_font_color="#22c55e",
                )
                
                # Update layout
                x_label = "Bi-week" if time_period == "Bi-weekly" else "Month"
                model_name = "Prediction Model"
                
                fig_residuals.update_layout(
                    title=dict(
                        text=f"<b>Residuals Over Time — {model_name}</b>",
                        font=dict(size=14, color=TEXT_PRIMARY),
                        x=0.5,
                    ),
                    xaxis=dict(
                        title=dict(text=f"<b>{x_label}</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                        tickfont=dict(size=10, color=TEXT_PRIMARY),
                        gridcolor=BORDER_COLOR,
                        gridwidth=1,
                        tickangle=-45,
                    ),
                    yaxis=dict(
                        title=dict(text="<b>Residual (Actual - Predicted) [%pts]</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                        tickfont=dict(size=10, color=TEXT_PRIMARY),
                        gridcolor=BORDER_COLOR,
                        gridwidth=1,
                    ),
                    plot_bgcolor=BG_CONTENT,
                    paper_bgcolor=BG_CONTENT,
                    font=dict(color=TEXT_PRIMARY),
                    hovermode="x unified",
                    legend=dict(
                        font=dict(size=10, color=TEXT_PRIMARY),
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor=BORDER_COLOR,
                        borderwidth=1,
                    ),
                    height=500,
                    margin=dict(l=50, r=50, t=60, b=80),
                )
                
                st.plotly_chart(fig_residuals, use_container_width=True, config={"displayModeBar": True})
                
                # Residual statistics
                st.markdown("### Residual Statistics")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric(
                        "Mean",
                        f"{resid_mean:.2f}%pts",
                        help="Average residual. Should be close to zero for unbiased predictions.",
                    )
                
                with stats_col2:
                    st.metric(
                        "Std Deviation",
                        f"{resid_std:.2f}%pts",
                        help="Standard deviation of residuals. Lower values indicate more consistent predictions.",
                    )
                
                with stats_col3:
                    st.metric(
                        "Minimum",
                        f"{resid_min:.2f}%pts",
                        help="Largest under-prediction (actual was much higher than predicted).",
                    )
                
                with stats_col4:
                    st.metric(
                        "Maximum",
                        f"{resid_max:.2f}%pts",
                        help="Largest over-prediction (actual was much lower than predicted).",
                    )
                
            else:
                st.warning(f"Required columns not found in predictions file. Expected 'Actual' and '{model_col}'.")
        else:
            st.warning(f"Predictions file not found: {predictions_path}")
    except Exception as e:
        st.error(f"Error loading residual analysis: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Future Forecast --------
    st.subheader("🔮 Future Forecast")
    
    st.markdown("""
    <div class="info-box" style="margin-bottom:1rem;">
        <p style="margin-bottom:0; font-size:0.9rem;">
            Forecast predictions for upcoming periods based on historical patterns and model predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Load predictions data for forecast
        if time_period == "Bi-weekly":
            predictions_path_forecast = Path("time-series-drug-adherence/data/test_predictions_all_models_biweekly.csv")
            model_col_forecast = "Prophet"
            forecast_periods = 2  # 2 upcoming weeks (2 bi-weeks)
            period_label = "Bi-week"
            date_offset = pd.DateOffset(weeks=2)
        else:
            predictions_path_forecast = Path("time-series-drug-adherence/data/test_predictions_all_models.csv")
            model_col_forecast = "XGBoost"
            forecast_periods = 1  # February (1 month)
            period_label = "Month"
            date_offset = pd.DateOffset(months=1)
        
        # Get last date from data
        last_date = data_ts.index[-1]
        
        # Generate forecast dates
        forecast_dates = []
        for i in range(forecast_periods):
            forecast_dates.append(last_date + date_offset * (i + 1))
        
        # Simple forecast based on recent trend and last prediction
        # Use the last predicted value from predictions file as baseline
        if predictions_path_forecast.exists():
            pred_df_forecast = pd.read_csv(predictions_path_forecast)
            date_col_forecast = "Month" if "Month" in pred_df_forecast.columns else "Date"
            pred_df_forecast[date_col_forecast] = pd.to_datetime(pred_df_forecast[date_col_forecast])
            
            if "Actual" in pred_df_forecast.columns and model_col_forecast in pred_df_forecast.columns:
                # Get last predicted value
                last_predicted = pred_df_forecast[model_col_forecast].iloc[-1] * 100 if pred_df_forecast[model_col_forecast].max() <= 1 else pred_df_forecast[model_col_forecast].iloc[-1]
                
                # Calculate trend from last few periods
                if len(pred_df_forecast) >= 3:
                    recent_actuals = pred_df_forecast["Actual"].iloc[-3:].values
                    recent_actuals = recent_actuals * 100 if recent_actuals.max() <= 1 else recent_actuals
                    trend = (recent_actuals[-1] - recent_actuals[0]) / len(recent_actuals)
                else:
                    trend = 0
                
                # Generate forecasts
                forecasts = []
                for i in range(forecast_periods):
                    # Simple forecast: last prediction + trend adjustment
                    forecast_val = last_predicted + (trend * (i + 1))
                    # Ensure forecast is within reasonable bounds
                    forecast_val = max(70, min(100, forecast_val))
                    forecasts.append(forecast_val)
                
                # Calculate confidence intervals (using historical std dev)
                if len(pred_df_forecast) > 1:
                    actual_values_forecast = pred_df_forecast["Actual"] * 100 if pred_df_forecast["Actual"].max() <= 1 else pred_df_forecast["Actual"]
                    hist_std = float(actual_values_forecast.std())
                else:
                    hist_std = 5.0  # Default std dev
                
                lower_ci = [f - 1.96 * hist_std for f in forecasts]
                upper_ci = [f + 1.96 * hist_std for f in forecasts]
                
                # Create forecast dataframe
                date_format = '%B %Y' if time_period == "Monthly" else '%b %d, %Y'
                forecast_df = pd.DataFrame({
                    period_label: [d.strftime(date_format) for d in forecast_dates],
                    "Predicted Adherence (%)": forecasts,
                    "Lower 95% CI": lower_ci,
                    "Upper 95% CI": upper_ci,
                })
                
                # Add status column
                def get_adherence_status_forecast(value):
                    if value >= 95:
                        return "🟢 Excellent"
                    elif value >= 85:
                        return "🟡 Good"
                    elif value >= 80:
                        return "🟠 Acceptable"
                    else:
                        return "🔴 Action Required"
                
                forecast_df["Status"] = forecast_df["Predicted Adherence (%)"].apply(get_adherence_status_forecast)
                
                # Display forecast table
                st.markdown("### Forecast Results")
                
                st.dataframe(
                    forecast_df.style.format({
                        "Predicted Adherence (%)": "{:.2f}",
                        "Lower 95% CI": "{:.2f}",
                        "Upper 95% CI": "{:.2f}",
                    }).background_gradient(
                        subset=["Predicted Adherence (%)"],
                        cmap="RdYlGn",
                        vmin=80,
                        vmax=95,
                    ),
                    use_container_width=True,
                )
                
                # Create forecast visualization
                st.markdown("### Forecast Visualization")
                
                fig_forecast = go.Figure()
                
                # Plot historical data (last 6 periods for context)
                hist_periods = min(6, len(data_ts))
                hist_dates = data_ts.index[-hist_periods:]
                hist_vals_series = data_ts[target_col].iloc[-hist_periods:]
                hist_values = hist_vals_series * 100 if hist_vals_series.max() <= 1 else hist_vals_series
                
                fig_forecast.add_trace(
                    go.Scatter(
                        x=hist_dates,
                        y=hist_values,
                        mode="lines+markers",
                        name="Historical",
                        line=dict(color="#38bdf8", width=2.5),
                        marker=dict(size=7, color="#38bdf8"),
                        hovertemplate="<b>%{x}</b><br>Historical: %{y:.2f}%<extra></extra>",
                    )
                )
                
                # Plot forecast
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecasts,
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#22c55e", width=3, dash="dash"),
                        marker=dict(size=10, color="#22c55e", symbol="diamond"),
                        hovertemplate="<b>%{x}</b><br>Forecast: %{y:.2f}%<extra></extra>",
                    )
                )
                
                # Add confidence interval
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_dates + forecast_dates[::-1],
                        y=upper_ci + lower_ci[::-1],
                        fill="toself",
                        fillcolor="rgba(34, 197, 94, 0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=True,
                        name="95% Confidence Interval",
                    )
                )
                
                # Add threshold lines
                fig_forecast.add_hline(
                    y=80,
                    line_dash="dot",
                    line_color=ACCENT_DANGER,
                    line_width=2,
                    opacity=0.5,
                    annotation_text="Minimum threshold (80%)",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color=ACCENT_DANGER,
                )
                
                fig_forecast.add_hline(
                    y=95,
                    line_dash="dot",
                    line_color="#22c55e",
                    line_width=2,
                    opacity=0.5,
                    annotation_text="Excellence threshold (95%)",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="#22c55e",
                )
                
                # Update layout
                x_label = "Bi-week" if time_period == "Bi-weekly" else "Month"
                forecast_title = f"February Forecast" if time_period == "Monthly" else "2-Week Forecast"
                
                fig_forecast.update_layout(
                    title=dict(
                        text=f"<b>{forecast_title} — {time_period}</b>",
                        font=dict(size=16, color=TEXT_PRIMARY),
                        x=0.5,
                    ),
                    xaxis=dict(
                        title=dict(text=f"<b>{x_label}</b>", font=dict(size=13, color=TEXT_PRIMARY)),
                        tickfont=dict(size=11, color=TEXT_PRIMARY),
                        gridcolor=BORDER_COLOR,
                        gridwidth=1,
                        tickangle=-45,
                    ),
                    yaxis=dict(
                        title=dict(text="<b>Refill adherence (%)</b>", font=dict(size=13, color=TEXT_PRIMARY)),
                        tickfont=dict(size=11, color=TEXT_PRIMARY),
                        gridcolor=BORDER_COLOR,
                        gridwidth=1,
                        range=[70, 100],
                    ),
                    plot_bgcolor=BG_CONTENT,
                    paper_bgcolor=BG_CONTENT,
                    font=dict(color=TEXT_PRIMARY),
                    hovermode="x unified",
                    legend=dict(
                        font=dict(size=11, color=TEXT_PRIMARY),
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor=BORDER_COLOR,
                        borderwidth=1,
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                    ),
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=100),
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": True})
                
                # Forecast alerts
                st.markdown("### 📊 Forecast Alerts")
                
                for idx, row in forecast_df.iterrows():
                    pred_val = row["Predicted Adherence (%)"]
                    period_name = row[period_label]
                    
                    if pred_val < 80:
                        st.error(f"⚠️ **{period_name}**: Predicted adherence ({pred_val:.2f}%) is **below 80%** threshold")
                    elif pred_val >= 95:
                        st.success(f"✓ **{period_name}**: Excellent adherence ({pred_val:.2f}%) - **above 95%**")
                    else:
                        st.info(f"✓ **{period_name}**: Good adherence ({pred_val:.2f}%) - within 80-95% range")
            else:
                st.warning("Unable to generate forecast: Required columns not found in predictions file.")
        else:
            st.warning("Unable to generate forecast: Predictions file not found.")
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        import traceback
        st.code(traceback.format_exc())

# -----------------------------------------------------------------------------
# DRUG CONSUMPTION FORECASTING MODULE
# -----------------------------------------------------------------------------
elif st.session_state["section"] == "consumption":
    st.header("💊 Drug Consumption Forecasting Workspace")
    st.subheader("📊 Monthly Drug Consumption Forecasts")

    st.markdown(
        """
        <div class="info-box">
            <h3 style="margin-bottom:4px;">Consumption forecasting scope</h3>
            <p style="margin-bottom:0;">Drug consumption is forecasted using <strong>RF+SARIMA Average</strong> models for the <strong>top 10 drugs</strong> (by ATC3 code) with monthly-level predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data loading functions
    @st.cache_data
    def load_consumption_data():
        """Load historical consumption data"""
        try:
            data_path = Path("tsfdc/data/atc3_monthly_full.csv")
            if not data_path.exists():
                return None
            df = pd.read_csv(data_path)
            df['month_start'] = pd.to_datetime(df['month_start'])
            return df
        except Exception as e:
            st.error(f"Error loading consumption data: {e}")
            return None

    @st.cache_data
    def load_forecast_data():
        """Load forecast data for top 10 drugs"""
        try:
            data_path = Path("tsfdc/forecast_plots/all_top10_forecasts.csv")
            if not data_path.exists():
                return None
            df = pd.read_csv(data_path)
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])
            df['last_observed_date'] = pd.to_datetime(df['last_observed_date'])
            return df
        except Exception as e:
            st.error(f"Error loading forecast data: {e}")
            return None

    def get_atc3_description(atc3_code):
        """Get drug description for ATC3 code"""
        atc3_descriptions = {
            "C09B": "ACE inhibitors and diuretics",
            "A02B": "Drugs for peptic ulcer and GERD",
            "C07A": "Beta blocking agents",
            "M01A": "Anti-inflammatory and antirheumatic",
            "N05B": "Anxiolytics",
            "B01A": "Antithrombotic agents",
            "C10A": "Lipid modifying agents",
            "C09A": "ACE inhibitors",
            "A10B": "Blood glucose lowering drugs",
            "N02A": "Opioids"
        }
        return atc3_descriptions.get(atc3_code, atc3_code)

    # Load data
    consumption_data = load_consumption_data()
    forecast_data = load_forecast_data()

    if consumption_data is None or forecast_data is None:
        st.error("Failed to load consumption or forecast data. Please check data files.")
        st.stop()

    # Filter to top 10 drugs from forecast data
    top10_atc3_codes = forecast_data['atc3_code'].unique().tolist()
    consumption_filtered = consumption_data[consumption_data['atc3_code'].isin(top10_atc3_codes)].copy()

    # Use Random Forest forecast directly (no averaging)
    # forecast_data already has 'rf_forecast' column, we'll use it directly

    # Load test predictions to calculate model metrics
    @st.cache_data
    def load_test_predictions():
        """Load test predictions for metrics calculation"""
        try:
            test_pred_path = Path("tsfdc/forecast_plots/test_predictions_all_top10.csv")
            if test_pred_path.exists():
                df = pd.read_csv(test_pred_path)
                df['month_start'] = pd.to_datetime(df['month_start'])
                return df
            return None
        except Exception:
            return None

    test_predictions_all = load_test_predictions()
    
    # Calculate aggregate metrics from test predictions
    model_mae = None
    model_rmse = None
    model_mape = None
    model_r2 = None
    
    if test_predictions_all is not None and len(test_predictions_all) > 0:
        # Filter out rows where actual is NaN or 0 (if needed)
        test_pred_valid = test_predictions_all.dropna(subset=['forecast', 'actual'])
        test_pred_valid = test_pred_valid[test_pred_valid['actual'] > 0]  # Avoid division by zero in MAPE
        
        if len(test_pred_valid) > 0:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            import numpy as np
            
            y_true = test_pred_valid['actual'].values
            y_pred = test_pred_valid['forecast'].values
            
            model_mae = mean_absolute_error(y_true, y_pred)
            model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            model_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            model_r2 = r2_score(y_true, y_pred)

    # -------- Executive Summary Panel --------
    st.subheader("📊 Executive Summary Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total drugs tracked",
            "10",
            help="Number of top drugs being forecasted",
        )

    with col2:
        total_forecast = forecast_data['rf_forecast'].sum()
        st.metric(
            "Total forecasted packages",
            f"{total_forecast:.0f}",
            help="Sum of forecasted packages across all top 10 drugs",
        )

    with col3:
        if model_mae is not None:
            st.metric(
                "Test MAE",
                f"{model_mae:.2f}",
                help="Mean Absolute Error on test set (lower is better)",
            )
        else:
            st.metric(
                "Test MAE",
                "N/A",
                help="Test metrics not available",
            )
    
    # Additional metrics row (only RMSE)
    if model_rmse is not None:
        st.markdown("<br/>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Test RMSE",
                f"{model_rmse:.2f}",
                help="Root Mean Squared Error on test set (lower is better)",
            )
        
        with col2:
            pass  # Empty column for spacing
        
        with col3:
            pass  # Empty column for spacing

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Top 10 Drugs Forecast Table --------
    st.subheader("📋 Top 10 Drugs Forecast Table")

    # Calculate average consumption for each drug from historical data
    avg_consumption = consumption_filtered.groupby('atc3_code')['packages'].mean().reset_index()
    avg_consumption.columns = ['atc3_code', 'avg_consumption']

    # Prepare forecast table
    forecast_table = forecast_data.copy()
    forecast_table['drug_name'] = forecast_table['atc3_code'].apply(get_atc3_description)
    
    # Merge average consumption
    forecast_table = forecast_table.merge(avg_consumption, on='atc3_code', how='left')
    
    # Calculate change percentage against average, handling zero values
    forecast_table['change_pct'] = forecast_table.apply(
        lambda row: ((row['rf_forecast'] - row['avg_consumption']) / 
                     (row['avg_consumption'] if row['avg_consumption'] != 0 else 1)) * 100,
        axis=1
    )
    
    def get_status_indicator(change_pct):
        if change_pct > 5:
            return "🟢 Increasing"
        elif change_pct < -5:
            return "🔴 Decreasing"
        else:
            return "🟡 Stable"

    forecast_table['status'] = forecast_table['change_pct'].apply(get_status_indicator)
    
    # Sort by forecast value
    forecast_table = forecast_table.sort_values('rf_forecast', ascending=False).reset_index(drop=True)
    
    # Create display table
    display_table = forecast_table[['drug_name', 'atc3_code', 'avg_consumption', 'rf_forecast', 'change_pct', 'status']].copy()
    display_table.columns = ['Drug Name', 'ATC3 Code', 'Average Consumption', 'Forecast', 'Change %', 'Status']
    display_table['Average Consumption'] = display_table['Average Consumption'].round(1)
    display_table['Forecast'] = display_table['Forecast'].round(1)
    display_table['Change %'] = display_table['Change %'].round(1)

    st.dataframe(
        display_table.style.format({
            'Average Consumption': '{:.1f}',
            'Forecast': '{:.1f}',
            'Change %': '{:.1f}%'
        }).background_gradient(
            subset=['Forecast'],
            cmap='YlOrRd',
            vmin=display_table['Forecast'].min(),
            vmax=display_table['Forecast'].max(),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Consumption Trends Visualization --------
    st.subheader("📈 Consumption Trends Visualization")

    # Prepare data for multi-line chart
    colors_professional = [
        "#38bdf8", "#22c55e", "#f97316", "#e879f9", "#a855f7",
        "#06b6d4", "#f59e0b", "#ef4444", "#10b981", "#6366f1"
    ]

    fig_trends = go.Figure()

    for idx, atc3_code in enumerate(top10_atc3_codes):
        drug_data = consumption_filtered[consumption_filtered['atc3_code'] == atc3_code].sort_values('month_start')
        drug_name = get_atc3_description(atc3_code)
        
        if len(drug_data) > 0:
            fig_trends.add_trace(
                go.Scatter(
                    x=drug_data['month_start'],
                    y=drug_data['packages'],
                    mode='lines+markers',
                    name=f"{atc3_code}: {drug_name}",
                    line=dict(color=colors_professional[idx % len(colors_professional)], width=2),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{atc3_code}: {drug_name}</b><br>Month: %{{x}}<br>Packages: %{{y}}<extra></extra>",
                )
            )

    # Add forecast points
    for idx, row in forecast_data.iterrows():
        atc3_code = row['atc3_code']
        drug_name = get_atc3_description(atc3_code)
        color_idx = top10_atc3_codes.index(atc3_code) % len(colors_professional)
        
        fig_trends.add_trace(
            go.Scatter(
                x=[row['forecast_date']],
                y=[row['rf_forecast']],
                mode='markers',
                name=f"{atc3_code} Forecast",
                marker=dict(
                    symbol='diamond',
                    size=12,
                    color=colors_professional[color_idx],
                    line=dict(width=2, color='#111827')
                ),
                hovertemplate=f"<b>{atc3_code}: {drug_name} Forecast</b><br>Month: %{{x}}<br>Forecast: %{{y:.1f}}<extra></extra>",
                showlegend=False,
            )
        )

    fig_trends.update_layout(
        title=dict(
            text="<b>Historical Consumption Trends — Top 10 Drugs</b>",
            font=dict(size=16, color=TEXT_PRIMARY),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="<b>Month</b>", font=dict(size=12, color=TEXT_PRIMARY)),
            tickfont=dict(size=11, color=TEXT_PRIMARY),
            gridcolor=BORDER_COLOR,
            gridwidth=1,
            tickangle=-45,
        ),
        yaxis=dict(
            title=dict(text="<b>Packages Consumed</b>", font=dict(size=12, color=TEXT_PRIMARY)),
            tickfont=dict(size=11, color=TEXT_PRIMARY),
            gridcolor=BORDER_COLOR,
            gridwidth=1,
        ),
        plot_bgcolor=BG_CONTENT,
        paper_bgcolor=BG_CONTENT,
        font=dict(color=TEXT_PRIMARY),
        hovermode='closest',
        legend=dict(
            font=dict(size=10, color=TEXT_PRIMARY),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        height=600,
        margin=dict(l=50, r=50, t=80, b=150),
    )

    st.plotly_chart(fig_trends, use_container_width=True, config={"displayModeBar": True})

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Individual Drug Drill-Down --------
    st.subheader("🔍 Individual Drug Drill-Down")

    selected_drug = st.selectbox(
        "Select a drug to view detailed analysis",
        options=top10_atc3_codes,
        format_func=lambda x: f"{x} — {get_atc3_description(x)}",
        key="drug_drilldown_select",
    )

    if selected_drug:
        drug_data = consumption_filtered[consumption_filtered['atc3_code'] == selected_drug].sort_values('month_start')
        drug_forecast = forecast_data[forecast_data['atc3_code'] == selected_drug].iloc[0]
        drug_name = get_atc3_description(selected_drug)

        st.markdown(
            f"""
            <div class="info-box" style="margin-bottom:1rem;">
                <h4 style="margin-bottom:4px;">Drug Information</h4>
                <p style="margin-bottom:2px;"><strong>ATC3 Code:</strong> {selected_drug}</p>
                <p style="margin-bottom:0;"><strong>Description:</strong> {drug_name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Determine train/test split at September 1, 2024 (matching notebook)
        # Exclude January 2025 from test period
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        test_predictions = None
        
        if len(drug_data) > 0:
            # Split at September 1, 2024, exclude January 2025 from test
            split_date = pd.Timestamp('2024-09-01')
            train_data = drug_data[drug_data['month_start'] < split_date].copy()
            test_data = drug_data[
                (drug_data['month_start'] >= split_date) & 
                (drug_data['month_start'] < pd.Timestamp('2025-01-01'))
            ].copy()
            
            # Try to load test predictions if available
            # First try combined file for all top 10 drugs (RF + SARIMA Average)
            combined_test_pred_path = Path("tsfdc/forecast_plots/test_predictions_all_top10.csv")
            # Fallback to old XGBoost file name for backward compatibility
            if not combined_test_pred_path.exists():
                combined_test_pred_path = Path("tsfdc/forecast_plots/xgb_test_predictions_all_top10.csv")
            
            test_predictions = None
            
            if combined_test_pred_path.exists():
                try:
                    test_pred_df = pd.read_csv(combined_test_pred_path)
                    # Filter by selected drug
                    if 'atc3_code' in test_pred_df.columns:
                        test_pred_df = test_pred_df[test_pred_df['atc3_code'] == selected_drug].copy()
                    
                    if len(test_pred_df) > 0:
                        if 'month_start' in test_pred_df.columns:
                            test_pred_df['month_start'] = pd.to_datetime(test_pred_df['month_start'])
                        elif 'Month' in test_pred_df.columns:
                            test_pred_df['Month'] = pd.to_datetime(test_pred_df['Month'])
                            test_pred_df = test_pred_df.rename(columns={'Month': 'month_start'})
                        elif 'Date' in test_pred_df.columns:
                            test_pred_df['Date'] = pd.to_datetime(test_pred_df['Date'])
                            test_pred_df = test_pred_df.rename(columns={'Date': 'month_start'})
                        
                        # Find prediction column (prioritize 'forecast' for RF+SARIMA avg, then fallback to others)
                        pred_col = None
                        for col in ['forecast', 'XGBoost', 'xgb_forecast', 'predicted', 'prediction', 'xgb_pred']:
                            if col in test_pred_df.columns:
                                pred_col = col
                                break
                        
                        if pred_col:
                            test_predictions = test_pred_df[['month_start', pred_col]].copy()
                            test_predictions.columns = ['month_start', 'prediction']
                            test_predictions = test_predictions.sort_values('month_start')
                except Exception as e:
                    test_predictions = None
                    import sys
                    print(f"Error loading combined test predictions: {e}", file=sys.stderr)
            
            # Fallback to individual file if combined file doesn't exist or didn't work
            if test_predictions is None or len(test_predictions) == 0:
                test_pred_path = Path(f"tsfdc/forecast_plots/test_predictions_{selected_drug}.csv")
                if not test_pred_path.exists():
                    # Try alternative naming (old XGBoost files)
                    test_pred_path = Path(f"tsfdc/forecast_plots/xgb_test_predictions_{selected_drug}.csv")
                
                if test_pred_path.exists():
                    try:
                        test_pred_df = pd.read_csv(test_pred_path)
                        if 'month_start' in test_pred_df.columns:
                            test_pred_df['month_start'] = pd.to_datetime(test_pred_df['month_start'])
                        elif 'Month' in test_pred_df.columns:
                            test_pred_df['Month'] = pd.to_datetime(test_pred_df['Month'])
                            test_pred_df = test_pred_df.rename(columns={'Month': 'month_start'})
                        elif 'Date' in test_pred_df.columns:
                            test_pred_df['Date'] = pd.to_datetime(test_pred_df['Date'])
                            test_pred_df = test_pred_df.rename(columns={'Date': 'month_start'})
                        
                        # Find prediction column (XGBoost, xgb_forecast, or similar)
                        pred_col = None
                        for col in ['XGBoost', 'xgb_forecast', 'predicted', 'prediction', 'forecast', 'xgb_pred']:
                            if col in test_pred_df.columns:
                                pred_col = col
                                break
                        
                        if pred_col:
                            test_predictions = test_pred_df[['month_start', pred_col]].copy()
                            test_predictions.columns = ['month_start', 'prediction']
                            test_predictions = test_predictions.sort_values('month_start')
                    except Exception as e:
                        test_predictions = None
                        import sys
                        print(f"Error loading individual test predictions: {e}", file=sys.stderr)

        # Show status if test predictions are available
        if test_predictions is not None and len(test_predictions) > 0:
            st.info(f"✓ Test predictions loaded (Forecasting): {len(test_predictions)} months available for comparison")

        # Detailed chart
        fig_drug = go.Figure()

        # Training data (historical consumption before test period)
        if len(train_data) > 0:
            fig_drug.add_trace(
                go.Scatter(
                    x=train_data['month_start'],
                    y=train_data['packages'],
                    mode='lines+markers',
                    name='Training Data',
                    line=dict(color=ACCENT_PRIMARY, width=3),
                    marker=dict(size=6, color=ACCENT_PRIMARY),
                    hovertemplate="<b>Training</b><br>Month: %{x}<br>Packages: %{y}<extra></extra>",
                )
            )

        # Test data (actual values)
        if len(test_data) > 0:
            fig_drug.add_trace(
                go.Scatter(
                    x=test_data['month_start'],
                    y=test_data['packages'],
                    mode='lines+markers',
                    name='Test Period (Actual)',
                    line=dict(color=ACCENT_WARNING, width=3, dash='dot'),
                    marker=dict(size=8, color=ACCENT_WARNING, symbol='square'),
                    hovertemplate="<b>Test Actual</b><br>Month: %{x}<br>Packages: %{y}<extra></extra>",
                )
            )

        # Test predictions line (Forecasting - green line)
        if test_predictions is not None and len(test_predictions) > 0:
            fig_drug.add_trace(
                go.Scatter(
                    x=test_predictions['month_start'],
                    y=test_predictions['prediction'],
                    mode='lines+markers',
                    name='Test Predictions (Forecasting)',
                    line=dict(color=ACCENT_SECONDARY, width=3, dash='dash'),
                    marker=dict(size=8, color=ACCENT_SECONDARY, symbol='triangle-up'),
                    hovertemplate="<b>Test Prediction (Forecasting)</b><br>Month: %{x}<br>Predicted: %{y:.1f}<extra></extra>",
                )
            )

        # Forecast point (future prediction)
        fig_drug.add_trace(
            go.Scatter(
                x=[drug_forecast['forecast_date']],
                y=[drug_forecast['rf_forecast']],
                mode='markers',
                name='Future Forecast (Forecasting)',
                marker=dict(symbol='diamond', size=20, color=ACCENT_SECONDARY, line=dict(width=3, color='#111827')),
                hovertemplate=f"<b>Future Forecast</b><br>Month: %{{x}}<br>Forecast: %{{y:.1f}}<extra></extra>",
            )
        )

        # Add train/test split line at September 1, 2024 (matching notebook test period start)
        split_date = pd.Timestamp('2024-09-01')
        
        # Get y-axis range for the line
        all_y_values = []
        if len(train_data) > 0:
            all_y_values.extend(train_data['packages'].tolist())
        if len(test_data) > 0:
            all_y_values.extend(test_data['packages'].tolist())
        if test_predictions is not None and len(test_predictions) > 0:
            all_y_values.extend(test_predictions['prediction'].tolist())
        if len(drug_data) > 0:
            all_y_values.extend(drug_data['packages'].tolist())
        
        if all_y_values:
            y_min = min(all_y_values) * 0.9
            y_max = max(all_y_values) * 1.1
            
            fig_drug.add_shape(
                type="line",
                x0=split_date,
                x1=split_date,
                y0=y_min,
                y1=y_max,
                line=dict(color="#ef4444", width=2, dash="dash"),
            )
            fig_drug.add_annotation(
                x=split_date,
                y=y_max,
                text="Train/Test Split (Sep 1, 2024)",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=10, color="#ef4444"),
                bgcolor="rgba(239, 68, 68, 0.1)",
                bordercolor="#ef4444",
                borderwidth=1,
            )

        fig_drug.update_layout(
            title=dict(
                text=f"<b>Consumption Trend — {selected_drug} ({drug_name})</b>",
                font=dict(size=16, color=TEXT_PRIMARY),
                x=0.5,
            ),
            xaxis=dict(
                title=dict(text="<b>Month</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                tickfont=dict(size=11, color=TEXT_PRIMARY),
                gridcolor=BORDER_COLOR,
                gridwidth=1,
                tickangle=-45,
            ),
            yaxis=dict(
                title=dict(text="<b>Packages Consumed</b>", font=dict(size=12, color=TEXT_PRIMARY)),
                tickfont=dict(size=11, color=TEXT_PRIMARY),
                gridcolor=BORDER_COLOR,
                gridwidth=1,
            ),
            plot_bgcolor=BG_CONTENT,
            paper_bgcolor=BG_CONTENT,
            font=dict(color=TEXT_PRIMARY),
            hovermode='closest',
            legend=dict(
                font=dict(size=11, color=TEXT_PRIMARY),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER_COLOR,
                borderwidth=1,
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=100),
        )

        st.plotly_chart(fig_drug, use_container_width=True, config={"displayModeBar": True})

        # Model performance metrics (if available) - Calculate from test predictions
        if test_predictions is not None and len(test_predictions) > 0:
            try:
                # Merge test predictions with actual values
                test_pred_with_actual = test_predictions.merge(
                    test_data[['month_start', 'packages']], 
                    on='month_start', 
                    how='inner'
                )
                test_pred_with_actual = test_pred_with_actual.rename(columns={'packages': 'actual'})
                
                # Filter out rows where actual is 0 or NaN (for MAPE calculation)
                test_pred_valid = test_pred_with_actual[
                    (test_pred_with_actual['actual'] > 0) & 
                    (test_pred_with_actual['prediction'].notna())
                ].copy()
                
                if len(test_pred_valid) > 0:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    import numpy as np
                    
                    y_true = test_pred_valid['actual'].values
                    y_pred = test_pred_valid['prediction'].values
                    
                    test_mae = mean_absolute_error(y_true, y_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    test_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    
                    st.markdown("### Model Performance Metrics (Forecasting)")
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Test MAE", f"{test_mae:.2f}")
                    with perf_col2:
                        st.metric("Test RMSE", f"{test_rmse:.2f}")
                    with perf_col3:
                        st.metric("Test MAPE", f"{test_mape:.2f}%")
            except Exception as e:
                pass  # Silently fail if metrics can't be calculated

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Forecast Comparison Chart --------
    st.subheader("📊 Forecast Comparison Chart")

    # Merge average consumption into forecast data
    forecast_sorted = forecast_data.merge(avg_consumption, on='atc3_code', how='left')
    forecast_sorted = forecast_sorted.sort_values('rf_forecast', ascending=True).copy()
    
    # Color code by change direction (against average)
    bar_colors = []
    for _, row in forecast_sorted.iterrows():
        avg_val = row['avg_consumption'] if row['avg_consumption'] != 0 else 1
        change_pct = ((row['rf_forecast'] - row['avg_consumption']) / avg_val) * 100
        if change_pct > 5:
            bar_colors.append(ACCENT_SECONDARY)  # Green
        elif change_pct < -5:
            bar_colors.append(ACCENT_DANGER)  # Red
        else:
            bar_colors.append(ACCENT_WARNING)  # Yellow

    fig_comparison = go.Figure()

    # Forecast bars
    fig_comparison.add_trace(
        go.Bar(
            x=[get_atc3_description(code) for code in forecast_sorted['atc3_code']],
            y=forecast_sorted['rf_forecast'],
            name='Ensemble Forecast',
            marker=dict(color=bar_colors, line=dict(color='#111827', width=1.5)),
            text=[f"{val:.1f}" for val in forecast_sorted['rf_forecast']],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Forecast: %{y:.1f} packages<extra></extra>",
        )
    )

    # Average consumption as reference line
    fig_comparison.add_trace(
        go.Scatter(
            x=[get_atc3_description(code) for code in forecast_sorted['atc3_code']],
            y=forecast_sorted['avg_consumption'],
            mode='markers',
            name='Average Consumption',
            marker=dict(symbol='circle', size=10, color=TEXT_MUTED, line=dict(width=2, color='#111827')),
            hovertemplate="<b>%{x}</b><br>Average: %{y:.1f} packages<extra></extra>",
        )
    )

    fig_comparison.update_layout(
        title=dict(
            text="<b>Forecasted Consumption Comparison — Top 10 Drugs</b>",
            font=dict(size=16, color=TEXT_PRIMARY),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="<b>Drug</b>", font=dict(size=12, color=TEXT_PRIMARY)),
            tickfont=dict(size=11, color=TEXT_PRIMARY),
            gridcolor=BORDER_COLOR,
            gridwidth=1,
            tickangle=-45,
        ),
        yaxis=dict(
            title=dict(text="<b>Packages</b>", font=dict(size=12, color=TEXT_PRIMARY)),
            tickfont=dict(size=11, color=TEXT_PRIMARY),
            gridcolor=BORDER_COLOR,
            gridwidth=1,
        ),
        plot_bgcolor=BG_CONTENT,
        paper_bgcolor=BG_CONTENT,
        font=dict(color=TEXT_PRIMARY),
        hovermode='closest',
        barmode='overlay',
        legend=dict(
            font=dict(size=11, color=TEXT_PRIMARY),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=150),
    )

    st.plotly_chart(fig_comparison, use_container_width=True, config={"displayModeBar": True})

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Model Performance Dashboard --------
    st.subheader("🤖 Model Performance Dashboard")

    # Calculate aggregate performance metrics from test predictions
    if test_predictions_all is not None and len(test_predictions_all) > 0:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        # Calculate metrics per drug
        all_metrics = []
        for atc3_code in top10_atc3_codes:
            drug_test_pred = test_predictions_all[test_predictions_all['atc3_code'] == atc3_code].copy()
            drug_test_pred = drug_test_pred.dropna(subset=['forecast', 'actual'])
            drug_test_pred = drug_test_pred[drug_test_pred['actual'] > 0]  # Avoid division by zero
            
            if len(drug_test_pred) > 0:
                y_true = drug_test_pred['actual'].values
                y_pred = drug_test_pred['forecast'].values
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                all_metrics.append({
                    'atc3_code': atc3_code,
                    'drug_name': get_atc3_description(atc3_code),
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                })

        if all_metrics:
            metrics_df_agg = pd.DataFrame(all_metrics)
            
            st.markdown("### Ensemble Model Performance Summary")
            
            agg_col1, agg_col2 = st.columns(2)
            
            with agg_col1:
                avg_mae = metrics_df_agg['MAE'].mean()
                st.metric("Average MAE", f"{avg_mae:.2f}", help="Mean Absolute Error across all drugs")
            
            with agg_col2:
                avg_rmse = metrics_df_agg['RMSE'].mean()
                st.metric("Average RMSE", f"{avg_rmse:.2f}", help="Root Mean Squared Error across all drugs")

            # Performance by drug table
            st.markdown("### Performance Metrics by Drug")
            
            display_metrics = metrics_df_agg[['drug_name', 'atc3_code', 'MAE', 'RMSE']].copy()
            display_metrics.columns = ['Drug Name', 'ATC3 Code', 'MAE', 'RMSE']
            display_metrics = display_metrics.sort_values('MAE', ascending=True).reset_index(drop=True)
            display_metrics['MAE'] = display_metrics['MAE'].round(2)
            display_metrics['RMSE'] = display_metrics['RMSE'].round(2)
            
            st.dataframe(
                display_metrics.style.format({
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}'
                }).background_gradient(
                    subset=['MAE'],
                    cmap='RdYlGn_r',  # Reversed: green is better (lower MAE)
                    vmin=display_metrics['MAE'].min(),
                    vmax=display_metrics['MAE'].max(),
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Model performance metrics are not available. Test predictions data is required to calculate metrics.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------- Consumption Heatmap/Comparison --------
    st.subheader("🔥 Consumption Heatmap")

    # Prepare heatmap data
    heatmap_data = consumption_filtered.pivot_table(
        index='atc3_code',
        columns='month_start',
        values='packages',
        aggfunc='sum'
    )

    # Sort by ATC3 code order from top 10
    heatmap_data = heatmap_data.reindex([code for code in top10_atc3_codes if code in heatmap_data.index])

    # Create heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(col)[:7] for col in heatmap_data.columns],  # Format dates
        y=[f"{idx} — {get_atc3_description(idx)}" for idx in heatmap_data.index],
        colorscale='YlOrRd',
        colorbar=dict(
            title=dict(
                text="Packages",
                font=dict(size=12, color=TEXT_PRIMARY),
            ),
            tickfont=dict(size=10, color=TEXT_PRIMARY),
        ),
        hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Packages: %{z}<extra></extra>",
    ))

    fig_heatmap.update_layout(
        title=dict(
            text="<b>Consumption Heatmap — Top 10 Drugs Over Time</b>",
            font=dict(size=16, color=TEXT_PRIMARY),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="<b>Month</b>", font=dict(size=12, color=TEXT_PRIMARY)),
            tickfont=dict(size=10, color=TEXT_PRIMARY),
            tickangle=-45,
        ),
        yaxis=dict(
            title=dict(text="<b>Drug (ATC3 Code)</b>", font=dict(size=12, color=TEXT_PRIMARY)),
            tickfont=dict(size=10, color=TEXT_PRIMARY),
        ),
        plot_bgcolor=BG_CONTENT,
        paper_bgcolor=BG_CONTENT,
        font=dict(color=TEXT_PRIMARY),
        height=600,
        margin=dict(l=200, r=50, t=80, b=150),
    )

    st.plotly_chart(fig_heatmap, use_container_width=True, config={"displayModeBar": True})

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='text-align: center; padding: 10px; background-color: {CARD_BG_SOFT}; border-radius: 10px; border: 1px solid {BORDER_COLOR};'>
        <p class="footer-text">
            <strong style='color:#e5e7eb;'>Healthcare Analytics Dashboard</strong><br/>
            Patient segmentation & drug adherence forecasting · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
