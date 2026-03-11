# =============================================================
#  REAL ESTATE INVESTMENT ADVISOR — app.py
#  Author : KAVYA S  |  Light Theme
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Light Theme CSS ───────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
.stApp, body, [data-testid="stAppViewContainer"] {
    background-color: #f8f9fa !important;
    color: #1a1a2e !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid #e0e0e0;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }

/* ── Section Header ── */
.sec-head {
    font-size: 1.25rem; font-weight: 700; color: #1a1a2e;
    border-left: 4px solid #2563eb; padding-left: 12px;
    margin: 22px 0 14px 0;
}

/* ── Metric Card ── */
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 22px 16px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-value { font-size: 1.9rem; font-weight: 800; }
.metric-label { font-size: 0.82rem; color: #64748b; margin-top: 4px; }

/* ── Info Box ── */
.info-box {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 20px;
    margin: 8px 0; color: #1a1a2e;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Row item ── */
.row-item {
    display: flex; justify-content: space-between; align-items: center;
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 10px 14px; margin: 5px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Step item ── */
.step-row {
    display: flex; align-items: center; gap: 12px; margin: 7px 0;
}
.step-num {
    background: #2563eb; color: #fff; font-weight: 800;
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.82rem; flex-shrink: 0;
}

/* ── Result cards ── */
.good-card {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #16a34a; border-radius: 16px;
    padding: 28px; text-align: center;
}
.bad-card {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #dc2626; border-radius: 16px;
    padding: 28px; text-align: center;
}
.price-card {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 2px solid #2563eb; border-radius: 16px;
    padding: 28px; text-align: center;
}

/* ── Best model banner ── */
.best-clf {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #16a34a; border-radius: 14px;
    padding: 20px; text-align: center;
}
.best-reg {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 2px solid #2563eb; border-radius: 14px;
    padding: 20px; text-align: center;
}

/* ── Skill badge ── */
.skill-badge {
    display: inline-block;
    background: #eff6ff; border: 1px solid #2563eb;
    color: #1d4ed8; padding: 5px 13px;
    border-radius: 20px; font-size: 0.82rem; margin: 3px;
}

/* ── Contact card ── */
.contact-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 13px 18px; margin: 6px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Divider ── */
.cdivider { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* ── Button ── */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white !important; border: none; border-radius: 10px;
    padding: 14px 40px; font-size: 1.05rem; font-weight: 700;
    width: 100%; transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* ── Inputs light ── */
.stSelectbox label, .stNumberInput label,
.stSlider label, .stRadio label {
    color: #1a1a2e !important; font-weight: 500;
}
div[data-baseweb="select"] > div {
    background: #ffffff !important; border: 1px solid #cbd5e1 !important;
    color: #1a1a2e !important;
}
input[type="number"] {
    background: #ffffff !important; color: #1a1a2e !important;
}

/* ── Dataframe ── */
.stDataFrame { background: #ffffff; }

/* ── Footer ── */
.footer {
    text-align: center; color: #64748b; font-size: 0.8rem;
    border-top: 1px solid #e2e8f0; padding-top: 18px; margin-top: 30px;
}

/* ── Objective row ── */
.obj-row {
    display: flex; align-items: center; gap: 12px;
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 10px 14px; margin: 5px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (
        joblib.load('models/best_classifier.pkl'),
        joblib.load('models/best_regressor.pkl'),
        joblib.load('models/label_encoders.pkl'),
        joblib.load('models/scaler.pkl'),
        joblib.load('models/feature_names.pkl'),
        joblib.load('models/model_info.pkl'),
    )

@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_data.csv')

@st.cache_data
def load_raw():
    return pd.read_csv('data/india_housing_prices.csv')

clf, reg, le_dict, scaler, feature_names, model_info = load_models()
df     = load_data()
raw_df = load_raw()

# ── Plot style ────────────────────────────────────────────────
def light_style():
    plt.rcParams.update({
        'figure.facecolor' : '#ffffff',
        'axes.facecolor'   : '#f8f9fa',
        'text.color'       : '#1a1a2e',
        'axes.labelcolor'  : '#1a1a2e',
        'xtick.color'      : '#475569',
        'ytick.color'      : '#475569',
        'axes.edgecolor'   : '#e2e8f0',
        'grid.color'       : '#e2e8f0',
        'axes.titlecolor'  : '#1a1a2e',
        'axes.titleweight' : 'bold',
        'axes.titlesize'   : 12,
    })

BLUE   = '#2563eb'
GREEN  = '#16a34a'
RED    = '#dc2626'
AMBER  = '#d97706'
PURPLE = '#7c3aed'
TEAL   = '#0891b2'
PAL5   = [BLUE, GREEN, AMBER, RED, PURPLE]

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 20px 0;'>
        <div style='font-size:2.4rem;'>🏠</div>
        <div style='font-size:1.05rem;font-weight:800;color:#ffffff;'>Real Estate</div>
        <div style='font-size:0.82rem;color:#93c5fd;font-weight:600;'>Investment Advisor</div>
    </div>
    <hr style='border-color:#334155;margin:0 0 14px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠 Home",
        "🗂️ View & Filter Data",
        "🔍 Predict Investment",
        "📊 Data Insights",
        "🔬 EDA Visualizations",
        "🤖 Model Performance",
        "👩‍💻 About Creator"
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#334155;margin:16px 0 10px 0;'>
    <div style='font-size:0.74rem;color:#94a3b8;text-align:center;'>
        Built with ❤️ by <b style='color:#93c5fd'>Kavya S</b><br>
        XGBoost · Streamlit · MLflow
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
#  PAGE 1 — HOME
# ==============================================================
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align:center;padding:28px 0 8px 0;'>
        <span style='font-size:3rem;'>🏠</span>
        <h1 style='font-size:2.4rem;font-weight:900;color:#1a1a2e;margin:6px 0 4px 0;'>
            Real Estate Investment Advisor
        </h1>
        <p style='font-size:1rem;color:#64748b;'>
            AI-powered Property Intelligence · Smart Investment Decisions
        </p>
    </div>
    <hr class='cdivider'>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,label,color) in zip([c1,c2,c3,c4],[
        ("2,50,000", "Properties Analysed", BLUE),
        ("99.81%",   "Classification Accuracy", GREEN),
        ("R² = 1.0", "Regression Score", AMBER),
        ("10",       "ML Models Trained", PURPLE),
    ]):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{color};'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-head'>🎯 Purpose</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='color:#374151;line-height:1.75;'>
            This application helps real estate investors make
            <b style='color:#2563eb;'>data-driven decisions</b> by leveraging
            Machine Learning on 2,50,000 Indian property listings.<br><br>
            Get <b style='color:#16a34a;'>instant AI predictions</b> on whether a
            property is worth buying and what it will be worth in 5 years.
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-head'>✅ Objectives</div>", unsafe_allow_html=True)
        for icon, text in [
            ("🏷️", "Classify whether a property is a <b>Good Investment</b>"),
            ("📈", "Predict <b>Future Price after 5 Years</b>"),
            ("📊", "Explore market trends with <b>interactive charts</b>"),
            ("🤖", "Compare <b>10 ML models</b> side by side"),
            ("💡", "Help investors make <b>smart, data-backed decisions</b>"),
        ]:
            st.markdown(f"""
            <div class='obj-row'>
                <span style='font-size:1.2rem;'>{icon}</span>
                <span style='font-size:0.9rem;color:#374151;'>{text}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='sec-head'>📁 Dataset Information</div>", unsafe_allow_html=True)
        for icon,key,val,color in [
            ("📄","Dataset",        "india_housing_prices.csv", BLUE),
            ("📏","Total Rows",     "2,50,000 properties",      GREEN),
            ("📋","Features",       "23 original + 6 engineered", AMBER),
            ("🏙️","Cities",         "42 cities across India",   PURPLE),
            ("🗺️","States",         "20 Indian states",         TEAL),
            ("🏠","Property Types", "Apartment · Villa · House", BLUE),
            ("🎯","Target 1",       "Good_Investment (0/1)",    GREEN),
            ("💰","Target 2",       "Future_Price_5Y (Lakhs)",  AMBER),
        ]:
            st.markdown(f"""
            <div class='row-item'>
                <span style='color:#64748b;font-size:0.88rem;'>{icon} {key}</span>
                <span style='color:{color};font-weight:600;font-size:0.88rem;'>{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-head'>🚀 How to Use</div>", unsafe_allow_html=True)
        for num,step in [
            ("1","Go to 🔍 Predict Investment page"),
            ("2","Fill in all property details"),
            ("3","Click the Predict Now button"),
            ("4","View investment verdict + future price"),
            ("5","Explore charts in 📊 Data Insights"),
        ]:
            st.markdown(f"""
            <div class='step-row'>
                <div class='step-num'>{num}</div>
                <span style='font-size:0.9rem;color:#374151;'>{step}</span>
            </div>""", unsafe_allow_html=True)



# ==============================================================
#  PAGE 2 — VIEW & FILTER DATA
# ==============================================================
elif page == "🗂️ View & Filter Data":

    st.markdown("""
    <h1 style='color:#1a1a2e;font-size:2rem;font-weight:800;'>🗂️ View & Filter Data</h1>
    <p style='color:#64748b;'>
        Browse the complete raw dataset first, then use dropdown filters below to narrow results.
    </p>
    <hr class='cdivider'>
    """, unsafe_allow_html=True)

    # ── overview cards ────────────────────────────────────────
    ov1, ov2, ov3, ov4, ov5 = st.columns(5)
    n_cat = len(raw_df.select_dtypes(include='object').columns)
    n_num = len(raw_df.select_dtypes(include=np.number).columns)
    for _c, (v, lbl, clr) in zip(
        [ov1, ov2, ov3, ov4, ov5],
        [
            (f"{len(raw_df):,}",  "Total Rows",       BLUE),
            (f"{raw_df.shape[1]}","Total Columns",    GREEN),
            (f"{n_cat}",          "Categorical Cols", PURPLE),
            (f"{n_num}",          "Numeric Cols",     AMBER),
            (f"{raw_df.isnull().sum().sum():,}", "Missing Values", RED),
        ]
    ):
        with _c:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{clr};font-size:1.5rem;'>{v}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 1 — COMPLETE RAW TABLE (shown first)
    # ══════════════════════════════════════════════════════════
    st.markdown("<div class='sec-head'>📊 Complete Dataset — india_housing_prices.csv</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-box' style='color:#374151;font-size:0.88rem;'>
        Showing the <b style='color:{BLUE};'>full unfiltered dataset</b> —
        <b>{len(raw_df):,} rows × {raw_df.shape[1]} columns</b>.
        Scroll down to use the <b style='color:#16a34a;'>Dropdown Filters</b>
        and see filtered results below.
    </div>""", unsafe_allow_html=True)

    st.dataframe(raw_df, use_container_width=True, height=500)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:0;border-top:2px dashed #cbd5e1;margin:8px 0 28px 0;'>",
        unsafe_allow_html=True
    )

    # ══════════════════════════════════════════════════════════
    #  SECTION 2 — ALL DROPDOWN FILTERS
    # ══════════════════════════════════════════════════════════
    st.markdown("<div class='sec-head'>🔽 Dropdown Filters</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='color:#374151;font-size:0.88rem;margin-bottom:18px;'>
        💡 Every control below is a <b>dropdown</b>. Choose values from any number of filters —
        they all combine with <b style='color:#2563eb;'>AND</b> logic.
        The filtered table appears at the bottom of this page.
    </div>""", unsafe_allow_html=True)

    filtered = raw_df.copy()

    # ── GROUP A: Categorical ──────────────────────────────────
    st.markdown("""
    <div style='font-size:0.92rem;font-weight:700;color:#2563eb;
                border-left:3px solid #2563eb;padding-left:10px;margin:14px 0 10px 0;'>
        🏷️ Categorical Filters
    </div>""", unsafe_allow_html=True)

    # Row 1 of categoricals
    ca1, ca2, ca3, ca4 = st.columns(4)
    cat_row1 = [
        ("State",            "🗺️ State"),
        ("City",             "🏙️ City"),
        ("Property_Type",    "🏠 Property Type"),
        ("Furnished_Status", "🛋️ Furnished Status"),
    ]
    for col_, (cname, label) in zip([ca1, ca2, ca3, ca4], cat_row1):
        with col_:
            opts = ["All"] + sorted(raw_df[cname].dropna().unique().tolist())
            sel  = st.selectbox(label, options=opts, key=f"dd_{cname}")
            if sel != "All":
                filtered = filtered[filtered[cname] == sel]

    # Row 2 of categoricals
    ca5, ca6, ca7, ca8 = st.columns(4)
    cat_row2 = [
        ("Facing",                         "🧭 Facing Direction"),
        ("Owner_Type",                     "👤 Owner Type"),
        ("Availability_Status",            "📌 Availability Status"),
        ("Public_Transport_Accessibility", "🚌 Public Transport"),
    ]
    for col_, (cname, label) in zip([ca5, ca6, ca7, ca8], cat_row2):
        with col_:
            opts = ["All"] + sorted(raw_df[cname].dropna().unique().tolist())
            sel  = st.selectbox(label, options=opts, key=f"dd_{cname}")
            if sel != "All":
                filtered = filtered[filtered[cname] == sel]

    # Locality + BHK row
    ca9, ca10, ca11, ca12 = st.columns(4)
    with ca9:
        loc_opts = ["All"] + sorted(raw_df["Locality"].dropna().unique().tolist())
        loc_sel  = st.selectbox("📍 Locality", options=loc_opts, key="dd_Locality")
        if loc_sel != "All":
            filtered = filtered[filtered["Locality"] == loc_sel]
    with ca10:
        bhk_opts = ["All"] + [str(x) for x in sorted(raw_df["BHK"].dropna().unique().tolist())]
        bhk_sel  = st.selectbox("🛏️ BHK", options=bhk_opts, key="dd_BHK")
        if bhk_sel != "All":
            filtered = filtered[filtered["BHK"] == int(bhk_sel)]
    with ca11:
        park_opts = ["All", "Yes (Has Parking)", "No (No Parking)"]
        park_sel  = st.selectbox("🚗 Parking Space", options=park_opts, key="dd_Parking")
        if "Yes" in park_sel:
            filtered = filtered[filtered["Parking_Space"] == 1]
        elif "No" in park_sel:
            filtered = filtered[filtered["Parking_Space"] == 0]
    with ca12:
        sec_opts  = ["All", "Yes (Has Security)", "No (No Security)"]
        sec_sel   = st.selectbox("🔒 Security", options=sec_opts, key="dd_Security")
        if "Yes" in sec_sel:
            filtered = filtered[filtered["Security"] == 1]
        elif "No" in sec_sel:
            filtered = filtered[filtered["Security"] == 0]

    # ── GROUP B: Numeric range dropdowns ─────────────────────
    st.markdown("""
    <div style='font-size:0.92rem;font-weight:700;color:#16a34a;
                border-left:3px solid #16a34a;padding-left:10px;margin:20px 0 10px 0;'>
        📊 Numeric Range Filters  (choose Min ≥ and Max ≤)
    </div>""", unsafe_allow_html=True)

    import math

    def make_buckets(series, n_buckets=10):
        """Return a list of evenly-spaced round bucket values for a numeric series."""
        lo, hi = float(series.min()), float(series.max())
        rng    = hi - lo
        if rng == 0:
            return [lo]
        raw_step = rng / n_buckets
        magnitude = 10 ** math.floor(math.log10(raw_step))
        nice_steps = [1, 2, 2.5, 5, 10]
        step = magnitude * min(nice_steps, key=lambda s: abs(s - raw_step / magnitude))
        start = math.floor(lo / step) * step
        end   = math.ceil(hi  / step) * step + step
        buckets = []
        v = start
        while v <= end + 1e-9:
            buckets.append(round(v, 6))
            v += step
        # clamp to actual data range
        buckets = [b for b in buckets if lo - step <= b <= hi + step]
        # always include true min/max
        if buckets[0] > lo:
            buckets.insert(0, round(lo, 2))
        if buckets[-1] < hi:
            buckets.append(round(hi, 2))
        return sorted(set(buckets))

    num_filter_defs = [
        ("Price_in_Lakhs",   "💰 Price (Lakhs)"),
        ("Size_in_SqFt",     "📐 Size (SqFt)"),
        ("Price_per_SqFt",   "💲 Price/SqFt"),
        ("Year_Built",       "📅 Year Built"),
        ("Floor_No",         "🏢 Floor No"),
        ("Total_Floors",     "🏗️ Total Floors"),
        ("Age_of_Property",  "⏳ Age (Years)"),
        ("Nearby_Schools",   "🏫 Nearby Schools"),
        ("Nearby_Hospitals", "🏥 Nearby Hospitals"),
    ]

    # render in rows of 3  (each col = 1 numeric filter with Min/Max dropdowns side by side)
    for row_start in range(0, len(num_filter_defs), 3):
        row_slice  = num_filter_defs[row_start:row_start + 3]
        row_cols   = st.columns(3)
        for ui_col, (cname, label) in zip(row_cols, row_slice):
            with ui_col:
                buckets = make_buckets(raw_df[cname])
                str_buckets = [str(int(b)) if b == int(b) else str(b) for b in buckets]

                st.markdown(
                    f"<div style='font-size:0.82rem;font-weight:600;color:#374151;"
                    f"margin-bottom:4px;'>{label}</div>",
                    unsafe_allow_html=True
                )
                mc1, mc2 = st.columns(2)
                with mc1:
                    sel_min = st.selectbox(
                        "≥ Min", options=["Any"] + str_buckets,
                        key=f"dd_min_{cname}", label_visibility="visible"
                    )
                with mc2:
                    sel_max = st.selectbox(
                        "≤ Max", options=["Any"] + str_buckets[::-1],
                        key=f"dd_max_{cname}", label_visibility="visible"
                    )

                if sel_min != "Any":
                    filtered = filtered[filtered[cname] >= float(sel_min)]
                if sel_max != "Any":
                    filtered = filtered[filtered[cname] <= float(sel_max)]

    # ── Filter result summary ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    total_r  = len(raw_df)
    filter_r = len(filtered)
    pct_kept = filter_r / total_r * 100 if total_r > 0 else 0

    sr1, sr2, sr3 = st.columns(3)
    for _c, (v, lbl, clr) in zip(
        [sr1, sr2, sr3],
        [
            (f"{filter_r:,}",           "Rows Matching",    BLUE),
            (f"{total_r - filter_r:,}", "Rows Filtered Out",RED),
            (f"{pct_kept:.1f}%",        "Data Retained",    GREEN),
        ]
    ):
        with _c:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{clr};font-size:1.5rem;'>{v}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Table controls (all dropdowns) ───────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-head'>📋 Filtered Table</div>", unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        sort_by  = st.selectbox(
            "↕️ Sort by column",
            options=["— No Sort —"] + raw_df.columns.tolist(),
            key="dd_sort_by"
        )
    with tc2:
        sort_dir = st.selectbox(
            "Sort direction",
            options=["Ascending ↑", "Descending ↓"],
            key="dd_sort_dir"
        )
    with tc3:
        n_rows = st.selectbox(
            "📄 Rows to show",
            options=[25, 50, 100, 250, 500, 1000, "All"],
            index=2,
            key="dd_n_rows"
        )

    disp_df   = filtered.copy()
    if sort_by != "— No Sort —":
        disp_df = disp_df.sort_values(by=sort_by, ascending=(sort_dir.startswith("Asc")))
    disp_show = disp_df if n_rows == "All" else disp_df.head(int(n_rows))

    st.markdown(f"""
    <div style='display:flex;gap:10px;align-items:center;margin:10px 0 8px 0;flex-wrap:wrap;'>
        <span style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:20px;
                     padding:4px 14px;font-size:0.82rem;color:#1d4ed8;font-weight:600;'>
            Showing {len(disp_show):,} of {filter_r:,} filtered rows
        </span>
        <span style='background:#f0fdf4;border:1px solid #86efac;border-radius:20px;
                     padding:4px 14px;font-size:0.82rem;color:#16a34a;font-weight:600;'>
            {raw_df.shape[1]} columns
        </span>
    </div>""", unsafe_allow_html=True)

    st.dataframe(disp_show, use_container_width=True, height=500)

    # ── Quick stats tabs ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-head'>📈 Quick Statistics (Filtered Data)</div>",
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs([
        "📊 Numeric Summary",
        "🏷️ Categorical Summary",
        "❓ Missing Values"
    ])

    with t1:
        num_c = raw_df.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(
            filtered[num_c].describe().T.round(2).reset_index()
            .rename(columns={"index": "Column"}),
            use_container_width=True, hide_index=True
        )

    with t2:
        cat_c = raw_df.select_dtypes(include='object').columns.tolist()
        rows  = []
        for c in cat_c:
            vc = filtered[c].value_counts()
            rows.append({
                "Column"        : c,
                "Unique Values" : int(filtered[c].nunique()),
                "Most Common"   : str(vc.index[0])  if len(vc) > 0 else "—",
                "Top Count"     : int(vc.iloc[0])   if len(vc) > 0 else 0,
                "Least Common"  : str(vc.index[-1]) if len(vc) > 0 else "—",
                "Least Count"   : int(vc.iloc[-1])  if len(vc) > 0 else 0,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with t3:
        all_c = raw_df.columns.tolist()
        miss_ = pd.DataFrame({
            "Column"        : all_c,
            "Missing Count" : [int(filtered[c].isnull().sum()) for c in all_c],
            "Missing %"     : [
                round(filtered[c].isnull().sum() / max(len(filtered), 1) * 100, 2)
                for c in all_c
            ],
            "Status"        : [
                "✅ Complete" if filtered[c].isnull().sum() == 0
                else f"⚠️ {filtered[c].isnull().sum():,} missing"
                for c in all_c
            ],
        })
        st.dataframe(miss_, use_container_width=True, hide_index=True)

    # ── Download buttons ──────────────────────────────────────
    st.markdown("<div class='sec-head'>💾 Export Filtered Data</div>",
                unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)

    with dl1:
        csv_b = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = f"⬇️ Download CSV  ({filter_r:,} rows)",
            data      = csv_b,
            file_name = "filtered_real_estate_data.csv",
            mime      = "text/csv",
            key       = "dl_csv_v3"
        )
    with dl2:
        import io
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            filtered.to_excel(writer, index=False, sheet_name="Filtered_Data")
        xbuf.seek(0)
        st.download_button(
            label     = f"⬇️ Download Excel ({filter_r:,} rows)",
            data      = xbuf.read(),
            file_name = "filtered_real_estate_data.xlsx",
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key       = "dl_excel_v3"
        )


# ==============================================================
#  PAGE 3 — PREDICT INVESTMENT
# ==============================================================
elif page == "🔍 Predict Investment":
    st.markdown("""
    <h1 style='color:#1a1a2e;font-size:2rem;font-weight:800;'>🔍 Predict Property Investment</h1>
    <p style='color:#64748b;'>Fill in the property details below to get an AI-powered prediction.</p>
    <hr class='cdivider'>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>📋 Enter Property Details</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        state         = st.selectbox("🗺️ State",           le_dict['State'].classes_)
        locality      = st.selectbox("📍 Locality",         le_dict['Locality'].classes_)
        bhk           = st.selectbox("🛏️ BHK",             [1,2,3,4,5])
        price         = st.number_input("💰 Price (Lakhs)", min_value=10.0, max_value=500.0, value=150.0, step=5.0)
        floor_no      = st.number_input("🏢 Floor Number",  min_value=0, max_value=30, value=5)
        schools       = st.slider("🏫 Nearby Schools",      1, 10, 5)
        furnished     = st.selectbox("🛋️ Furnished Status", le_dict['Furnished_Status'].classes_)
        owner         = st.selectbox("👤 Owner Type",       le_dict['Owner_Type'].classes_)

    with c2:
        city          = st.selectbox("🏙️ City",             le_dict['City'].classes_)
        prop_type     = st.selectbox("🏠 Property Type",    le_dict['Property_Type'].classes_)
        size          = st.number_input("📐 Size (SqFt)",   min_value=500, max_value=5000, value=1500, step=50)
        year_built    = st.number_input("📅 Year Built",    min_value=1990, max_value=2023, value=2010)
        total_floors  = st.number_input("🏗️ Total Floors",  min_value=1, max_value=30, value=10)
        hospitals     = st.slider("🏥 Nearby Hospitals",    1, 10, 5)
        facing        = st.selectbox("🧭 Facing Direction", le_dict['Facing'].classes_)
        availability  = st.selectbox("📌 Availability",     le_dict['Availability_Status'].classes_)

    with c3:
        transport     = st.selectbox("🚌 Public Transport", le_dict['Public_Transport_Accessibility'].classes_)
        amenity_count = st.slider("🏊 Amenity Count",       1, 5, 3)
        has_parking   = st.selectbox("🚗 Parking",          ["Yes","No"])
        has_security  = st.selectbox("🔒 Security",         ["Yes","No"])

        st.markdown("""
        <div class='info-box' style='margin-top:18px;color:#374151;'>
            <b style='color:#2563eb;'>⚙️ Auto-calculated features:</b><br>
            <span style='font-size:0.82rem;line-height:2;'>
            • Price per SqFt<br>• Age of Property<br>
            • Floor Ratio<br>• Amenity Density Score<br>• Is Ready to Move
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 PREDICT NOW")

    if predict_btn:
        price_per_sqft  = price / size
        age_of_property = 2024 - year_built
        floor_ratio     = floor_no / (total_floors + 1)
        amenity_density = amenity_count / (size / 1000)
        is_ready        = 1 if availability == 'Ready_to_Move' else 0
        parking_val     = 1 if has_parking  == 'Yes' else 0
        security_val    = 1 if has_security == 'Yes' else 0

        input_dict = {
            'State'                          : le_dict['State'].transform([state])[0],
            'City'                           : le_dict['City'].transform([city])[0],
            'Locality'                       : le_dict['Locality'].transform([locality])[0],
            'Property_Type'                  : le_dict['Property_Type'].transform([prop_type])[0],
            'BHK'                            : bhk,
            'Size_in_SqFt'                   : size,
            'Price_in_Lakhs'                 : price,
            'Price_per_SqFt'                 : price_per_sqft,
            'Year_Built'                     : year_built,
            'Furnished_Status'               : le_dict['Furnished_Status'].transform([furnished])[0],
            'Floor_No'                       : floor_no,
            'Total_Floors'                   : total_floors,
            'Age_of_Property'                : age_of_property,
            'Nearby_Schools'                 : schools,
            'Nearby_Hospitals'               : hospitals,
            'Public_Transport_Accessibility' : le_dict['Public_Transport_Accessibility'].transform([transport])[0],
            'Facing'                         : le_dict['Facing'].transform([facing])[0],
            'Owner_Type'                     : le_dict['Owner_Type'].transform([owner])[0],
            'Availability_Status'            : le_dict['Availability_Status'].transform([availability])[0],
            'Amenity_Count'                  : amenity_count,
            'Floor_Ratio'                    : floor_ratio,
            'Amenity_Density_Score'          : amenity_density,
            'Is_Ready_to_Move'               : is_ready,
            'Has_Parking'                    : parking_val,
            'Has_Security'                   : security_val,
        }

        input_df     = pd.DataFrame([input_dict])[feature_names]
        exclude_cols = ['Is_Ready_to_Move','Has_Parking','Has_Security']
        scale_cols   = [c for c in feature_names if c not in exclude_cols]
        input_scaled = input_df.copy()
        input_scaled[scale_cols] = scaler.transform(input_df[scale_cols])

        clf_pred     = clf.predict(input_scaled)[0]
        clf_proba    = clf.predict_proba(input_scaled)[0]
        confidence   = clf_proba[clf_pred] * 100
        future_price = reg.predict(input_scaled)[0]
        growth       = future_price - price
        growth_pct   = (growth / price) * 100

        st.markdown("<hr class='cdivider'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-head'>🎯 Prediction Results</div>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            if clf_pred == 1:
                st.markdown(f"""
                <div class='good-card'>
                    <div style='font-size:2.4rem;'>✅</div>
                    <div style='font-size:1.5rem;font-weight:800;color:#16a34a;margin:8px 0;'>
                        GOOD INVESTMENT!
                    </div>
                    <div style='color:#374151;font-size:0.95rem;'>
                        Confidence: <b style='color:#16a34a;font-size:1.3rem;'>{confidence:.1f}%</b>
                    </div>
                    <div style='color:#6b7280;font-size:0.8rem;margin-top:8px;'>
                        Model: {model_info['best_classifier_name']}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='bad-card'>
                    <div style='font-size:2.4rem;'>❌</div>
                    <div style='font-size:1.5rem;font-weight:800;color:#dc2626;margin:8px 0;'>
                        NOT A GOOD INVESTMENT
                    </div>
                    <div style='color:#374151;font-size:0.95rem;'>
                        Confidence: <b style='color:#dc2626;font-size:1.3rem;'>{confidence:.1f}%</b>
                    </div>
                    <div style='color:#6b7280;font-size:0.8rem;margin-top:8px;'>
                        Model: {model_info['best_classifier_name']}
                    </div>
                </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class='price-card'>
                <div style='font-size:2rem;'>📈</div>
                <div style='font-size:1.05rem;font-weight:700;color:#2563eb;margin:6px 0;'>
                    Estimated Price After 5 Years
                </div>
                <div style='font-size:2rem;font-weight:900;color:#1a1a2e;margin:8px 0;'>
                    ₹ {future_price:,.2f} Lakhs
                </div>
                <div style='display:flex;justify-content:center;gap:24px;margin-top:12px;'>
                    <div style='text-align:center;'>
                        <div style='color:#64748b;font-size:0.75rem;'>Current</div>
                        <div style='color:#1a1a2e;font-weight:700;'>₹{price:,.0f}L</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='color:#64748b;font-size:0.75rem;'>Growth</div>
                        <div style='color:#16a34a;font-weight:700;'>+₹{growth:,.1f}L</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='color:#64748b;font-size:0.75rem;'>Return</div>
                        <div style='color:#d97706;font-weight:700;'>{growth_pct:.1f}%</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        light_style()
        ch1, ch2 = st.columns(2)

        with ch1:
            fig, ax = plt.subplots(figsize=(6,3.5))
            vals  = [clf_proba[0]*100, clf_proba[1]*100]
            cats  = ['Not Good','Good Investment']
            clrs  = [RED, GREEN]
            bars  = ax.barh(cats, vals, color=clrs, edgecolor='#ffffff', height=0.4)
            for bar,v in zip(bars,vals):
                ax.text(v+1, bar.get_y()+bar.get_height()/2,
                        f'{v:.1f}%', va='center', color='#1a1a2e', fontweight='bold')
            ax.set_xlim(0,115)
            ax.set_xlabel('Confidence %')
            ax.set_title('Prediction Confidence')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.4, axis='x')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with ch2:
            years  = list(range(0,6))
            prices = [price*(1.08**y) for y in years]
            fig, ax = plt.subplots(figsize=(6,3.5))
            ax.plot(years, prices, color=BLUE, linewidth=2.5,
                    marker='o', markersize=8, markerfacecolor=AMBER,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.fill_between(years, prices, alpha=0.12, color=BLUE)
            for y,p in zip(years,prices):
                ax.text(y, p + (max(prices)-min(prices))*0.04,
                        f'₹{p:.0f}L', ha='center', fontsize=8, color='#1a1a2e', fontweight='bold')
            ax.set_xlabel('Year'); ax.set_ylabel('Price (Lakhs)')
            ax.set_title('Year-by-Year Price Growth')
            ax.set_xticks(years)
            ax.set_xticklabels([f'Y{y}' for y in years])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ==============================================================
#  PAGE 3 — DATA INSIGHTS
# ==============================================================
elif page == "📊 Data Insights":
    st.markdown("""
    <h1 style='color:#1a1a2e;font-size:2rem;font-weight:800;'>📊 Data Insights</h1>
    <p style='color:#64748b;'>Explore and filter the real estate dataset with interactive charts.</p>
    <hr class='cdivider'>
    """, unsafe_allow_html=True)

    df_d = df.copy()
    df_d['City_Label']  = le_dict['City'].inverse_transform(df_d['City'].astype(int))
    df_d['PType_Label'] = le_dict['Property_Type'].inverse_transform(df_d['Property_Type'].astype(int))

    st.markdown("<div class='sec-head'>🔽 Filters</div>", unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    with f1:
        sel_city = st.selectbox("🏙️ City", ['All']+sorted(df_d['City_Label'].unique().tolist()))
    with f2:
        sel_pt   = st.selectbox("🏠 Property Type", ['All']+sorted(df_d['PType_Label'].unique().tolist()))
    with f3:
        sel_bhk  = st.selectbox("🛏️ BHK", ['All']+sorted(df_d['BHK'].unique().tolist()))
    with f4:
        pmin,pmax = float(df_d['Price_in_Lakhs'].min()), float(df_d['Price_in_Lakhs'].max())
        price_range = st.slider("💰 Price Range (Lakhs)", pmin, pmax, (pmin,pmax))

    filtered = df_d.copy()
    if sel_city != 'All': filtered = filtered[filtered['City_Label']==sel_city]
    if sel_pt   != 'All': filtered = filtered[filtered['PType_Label']==sel_pt]
    if sel_bhk  != 'All': filtered = filtered[filtered['BHK']==sel_bhk]
    filtered = filtered[(filtered['Price_in_Lakhs']>=price_range[0])&
                        (filtered['Price_in_Lakhs']<=price_range[1])]

    st.markdown(f"<p style='color:{BLUE};font-size:0.9rem;'>📌 Showing <b>{len(filtered):,}</b> properties after filters</p>",
                unsafe_allow_html=True)

    light_style()

    g1,g2 = st.columns(2)
    with g1:
        st.markdown("<div class='sec-head'>💰 Price Distribution</div>", unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(7,4))
        ax.hist(filtered['Price_in_Lakhs'], bins=50, color=BLUE, edgecolor='white', alpha=0.8)
        ax.axvline(filtered['Price_in_Lakhs'].mean(),   color=AMBER, linestyle='--', linewidth=2,
                   label=f'Mean: ₹{filtered["Price_in_Lakhs"].mean():.0f}L')
        ax.axvline(filtered['Price_in_Lakhs'].median(), color=RED,   linestyle='--', linewidth=2,
                   label=f'Median: ₹{filtered["Price_in_Lakhs"].median():.0f}L')
        ax.set_xlabel('Price (Lakhs)'); ax.set_ylabel('Count')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with g2:
        st.markdown("<div class='sec-head'>🏙️ Avg Price by City (Top 12)</div>", unsafe_allow_html=True)
        city_avg = filtered.groupby('City_Label')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(12)
        fig,ax   = plt.subplots(figsize=(7,4))
        colors_  = [BLUE if i==0 else '#93c5fd' for i in range(len(city_avg))]
        ax.bar(city_avg.index, city_avg.values, color=colors_, edgecolor='white', width=0.65)
        ax.set_xlabel('City'); ax.set_ylabel('Avg Price (Lakhs)')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.4, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    g3,g4 = st.columns(2)
    with g3:
        st.markdown("<div class='sec-head'>🏠 Property Type Share</div>", unsafe_allow_html=True)
        pt_counts = filtered['PType_Label'].value_counts()
        fig,ax    = plt.subplots(figsize=(6,4))
        ax.pie(pt_counts.values, labels=pt_counts.index,
               autopct='%1.1f%%', colors=[BLUE, GREEN, AMBER],
               textprops={'color':'#1a1a2e','fontsize':11},
               wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('Property Type Distribution')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with g4:
        st.markdown("<div class='sec-head'>🛏️ BHK vs Price</div>", unsafe_allow_html=True)
        bhk_list = sorted(filtered['BHK'].unique())
        bhk_data = [filtered[filtered['BHK']==b]['Price_in_Lakhs'].dropna() for b in bhk_list]
        fig,ax   = plt.subplots(figsize=(7,4))
        bp = ax.boxplot(bhk_data, labels=[f'{b} BHK' for b in bhk_list],
                        patch_artist=True,
                        medianprops=dict(color=AMBER, linewidth=2.5),
                        whiskerprops=dict(color='#94a3b8'),
                        capprops=dict(color='#94a3b8'),
                        flierprops=dict(marker='o', alpha=0.3, markersize=3, color='#94a3b8'))
        for patch,c in zip(bp['boxes'], PAL5):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        ax.set_ylabel('Price (Lakhs)')
        ax.grid(True, alpha=0.4, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    g5,g6 = st.columns(2)
    with g5:
        st.markdown("<div class='sec-head'>✅ Good Investment % by City</div>", unsafe_allow_html=True)
        gi_city = filtered.groupby('City_Label')['Good_Investment'].mean().sort_values(ascending=False).head(12)*100
        fig,ax  = plt.subplots(figsize=(7,4))
        colors_ = [GREEN if v>=50 else RED for v in gi_city.values]
        ax.barh(gi_city.index, gi_city.values, color=colors_, edgecolor='white', height=0.55)
        ax.axvline(50, color='#94a3b8', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Good Investment %')
        ax.grid(True, alpha=0.4, axis='x')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with g6:
        st.markdown("<div class='sec-head'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
        num_cols = ['BHK','Size_in_SqFt','Price_in_Lakhs','Age_of_Property',
                    'Nearby_Schools','Nearby_Hospitals','Amenity_Count','Good_Investment']
        corr = filtered[num_cols].corr()
        fig,ax = plt.subplots(figsize=(7,5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fa')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    ax=ax, linewidths=0.5, annot_kws={'size':8,'color':'#1a1a2e'},
                    cbar_kws={'shrink':0.8})
        ax.tick_params(colors='#475569', labelsize=8)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ==============================================================
#  PAGE 4 — EDA VISUALIZATIONS
# ==============================================================
elif page == "🔬 EDA Visualizations":
    st.markdown("""
    <h1 style='color:#1a1a2e;font-size:2rem;font-weight:800;'>🔬 Exploratory Data Analysis</h1>
    <p style='color:#64748b;'>Select a group, then select a question — view its code and visualization.</p>
    <hr class='cdivider'>
    """, unsafe_allow_html=True)

    # ── decoded data ──────────────────────────────────────────
    eda = df.copy()
    eda['City_Label']      = le_dict['City'].inverse_transform(eda['City'].astype(int))
    eda['State_Label']     = le_dict['State'].inverse_transform(eda['State'].astype(int))
    eda['Locality_Label']  = le_dict['Locality'].inverse_transform(eda['Locality'].astype(int))
    eda['PType_Label']     = le_dict['Property_Type'].inverse_transform(eda['Property_Type'].astype(int))
    eda['Furnished_Label'] = le_dict['Furnished_Status'].inverse_transform(eda['Furnished_Status'].astype(int))
    eda['Facing_Label']    = le_dict['Facing'].inverse_transform(eda['Facing'].astype(int))
    eda['Owner_Label']     = le_dict['Owner_Type'].inverse_transform(eda['Owner_Type'].astype(int))
    eda['Avail_Label']     = le_dict['Availability_Status'].inverse_transform(eda['Availability_Status'].astype(int))
    eda['Transport_Label'] = le_dict['Public_Transport_Accessibility'].inverse_transform(eda['Public_Transport_Accessibility'].astype(int))

    EDA_GROUPS = {
        "📦 Q1–5: Price & Size Analysis": [
            "Q1. What is the distribution of property prices?",
            "Q2. What is the distribution of property sizes?",
            "Q3. How does the price per sq ft vary by property type?",
            "Q4. Is there a relationship between property size and price?",
            "Q5. Are there any outliers in price per sq ft or property size?",
        ],
        "📍 Q6–10: Location-based Analysis": [
            "Q6. What is the average price per sq ft by state?",
            "Q7. What is the average property price by city?",
            "Q8. What is the median age of properties by locality?",
            "Q9. How is BHK distributed across cities?",
            "Q10. What are the price trends for the top 5 most expensive localities?",
        ],
        "🔗 Q11–15: Feature Relationship & Correlation": [
            "Q11. How are numeric features correlated with each other?",
            "Q12. How do nearby schools relate to price per sq ft?",
            "Q13. How do nearby hospitals relate to price per sq ft?",
            "Q14. How does price vary by furnished status?",
            "Q15. How does price per sq ft vary by property facing direction?",
        ],
        "💡 Q16–20: Investment / Amenities / Ownership": [
            "Q16. How many properties belong to each owner type?",
            "Q17. How many properties are available under each availability status?",
            "Q18. Does parking space affect property price?",
            "Q19. How do amenities affect price per sq ft?",
            "Q20. How does public transport accessibility relate to price per sq ft or investment potential?",
        ],
    }

    # ── code snippets dict ────────────────────────────────────
    EDA_CODE = {
        "Q1": """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df['Price_in_Lakhs'], bins=60, color='#2563eb', edgecolor='white', alpha=0.85)
ax.axvline(df['Price_in_Lakhs'].mean(),   color='red',    linestyle='--', lw=2, label=f"Mean")
ax.axvline(df['Price_in_Lakhs'].median(), color='orange', linestyle='--', lw=2, label=f"Median")
ax.set_xlabel('Price (Lakhs)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Property Prices')
ax.legend()
plt.tight_layout()
plt.show()""",

        "Q2": """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df['Size_in_SqFt'], bins=60, color='#16a34a', edgecolor='white', alpha=0.85)
ax.axvline(df['Size_in_SqFt'].mean(),   color='red',    linestyle='--', lw=2, label=f"Mean")
ax.axvline(df['Size_in_SqFt'].median(), color='orange', linestyle='--', lw=2, label=f"Median")
ax.set_xlabel('Size (SqFt)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Property Sizes')
ax.legend()
plt.tight_layout()
plt.show()""",

        "Q3": """\
import matplotlib.pyplot as plt

pt_groups = [df[df['Property_Type']==p]['Price_per_SqFt'].dropna()
             for p in sorted(df['Property_Type'].unique())]
pt_labels = sorted(df['Property_Type'].unique())

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(pt_groups, labels=pt_labels, patch_artist=True)
ax.set_ylabel('Price per SqFt (₹)')
ax.set_title('Price per SqFt by Property Type')
plt.tight_layout()
plt.show()""",

        "Q4": """\
import matplotlib.pyplot as plt
import numpy as np

sample = df.sample(5000, random_state=42)
fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(sample['Size_in_SqFt'], sample['Price_in_Lakhs'],
                alpha=0.25, s=10, c=sample['BHK'], cmap='coolwarm')
plt.colorbar(sc, ax=ax, label='BHK')
m, b = np.polyfit(df['Size_in_SqFt'], df['Price_in_Lakhs'], 1)
xs   = np.linspace(df['Size_in_SqFt'].min(), df['Size_in_SqFt'].max(), 100)
ax.plot(xs, m*xs+b, color='red', linewidth=2, label='Trend Line')
ax.set_xlabel('Size (SqFt)')
ax.set_ylabel('Price (Lakhs)')
ax.set_title('Property Size vs Price')
ax.legend()
plt.tight_layout()
plt.show()""",

        "Q5": """\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, col in zip(axes, ['Price_per_SqFt', 'Size_in_SqFt']):
    ax.boxplot(df[col].dropna(), vert=False, patch_artist=True)
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)]
    ax.set_title(f'{col}\\n{len(outliers):,} outliers ({len(outliers)/len(df)*100:.1f}%)')
    ax.set_xlabel(col)
plt.tight_layout()
plt.show()""",

        "Q6": """\
import matplotlib.pyplot as plt

state_avg = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=True)
fig, ax   = plt.subplots(figsize=(14, 6))
ax.barh(state_avg.index, state_avg.values, color='#2563eb', edgecolor='white')
ax.axvline(state_avg.median(), color='orange', linestyle='--', lw=2, label='Median')
ax.set_xlabel('Avg Price per SqFt (₹)')
ax.set_title('Average Price per SqFt by State')
ax.legend()
plt.tight_layout()
plt.show()""",

        "Q7": """\
import matplotlib.pyplot as plt

city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15)
fig, ax  = plt.subplots(figsize=(12, 5))
ax.bar(city_avg.index, city_avg.values, color='#2563eb', edgecolor='white')
ax.set_xlabel('City')
ax.set_ylabel('Avg Price (Lakhs)')
ax.set_title('Average Property Price by City (Top 15)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()""",

        "Q8": """\
import matplotlib.pyplot as plt

loc_age = df.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(loc_age.index, loc_age.values, color='#d97706', edgecolor='white')
ax.set_xlabel('Median Age (Years)')
ax.set_title('Median Age of Properties by Locality (Top 20)')
plt.tight_layout()
plt.show()""",

        "Q9": """\
import matplotlib.pyplot as plt

top10_cities = df['City'].value_counts().head(10).index
bhk_city     = (df[df['City'].isin(top10_cities)]
                .groupby(['City','BHK']).size().unstack(fill_value=0))
fig, ax = plt.subplots(figsize=(14, 5))
bhk_city.plot(kind='bar', ax=ax, edgecolor='white', width=0.75)
ax.set_xlabel('City')
ax.set_ylabel('Count')
ax.set_title('BHK Distribution across Top 10 Cities')
ax.legend(title='BHK')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()""",

        "Q10": """\
import matplotlib.pyplot as plt

top5_loc = (df.groupby('Locality')['Price_in_Lakhs']
            .mean().sort_values(ascending=False).head(5).index)
fig, ax  = plt.subplots(figsize=(14, 4))
for loc in top5_loc:
    sub = df[df['Locality']==loc].groupby('Year_Built')['Price_in_Lakhs'].mean()
    ax.plot(sub.index, sub.values, marker='o', linewidth=2, label=loc)
ax.set_xlabel('Year Built')
ax.set_ylabel('Avg Price (Lakhs)')
ax.set_title('Price Trends — Top 5 Expensive Localities')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()""",

        "Q11": """\
import seaborn as sns
import matplotlib.pyplot as plt

num_cols = ['BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt',
            'Age_of_Property','Floor_No','Total_Floors',
            'Nearby_Schools','Nearby_Hospitals','Amenity_Count',
            'Floor_Ratio','Good_Investment','Future_Price_5Y']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax, linewidths=0.5)
ax.set_title('Numeric Feature Correlation Matrix')
plt.tight_layout()
plt.show()""",

        "Q12": """\
import matplotlib.pyplot as plt

school_avg = df.groupby('Nearby_Schools')['Price_per_SqFt'].mean()
fig, ax    = plt.subplots(figsize=(10, 5))
ax.bar(school_avg.index, school_avg.values, color='#2563eb', edgecolor='white', width=0.6)
ax.set_xlabel('Number of Nearby Schools')
ax.set_ylabel('Avg Price per SqFt (₹)')
ax.set_title('Nearby Schools vs Price per SqFt')
plt.tight_layout()
plt.show()""",

        "Q13": """\
import matplotlib.pyplot as plt

hosp_avg = df.groupby('Nearby_Hospitals')['Price_per_SqFt'].mean()
fig, ax  = plt.subplots(figsize=(10, 5))
ax.bar(hosp_avg.index, hosp_avg.values, color='#16a34a', edgecolor='white', width=0.6)
ax.set_xlabel('Number of Nearby Hospitals')
ax.set_ylabel('Avg Price per SqFt (₹)')
ax.set_title('Nearby Hospitals vs Price per SqFt')
plt.tight_layout()
plt.show()""",

        "Q14": """\
import matplotlib.pyplot as plt

furn_groups = [df[df['Furnished_Status']==f]['Price_in_Lakhs'].dropna()
               for f in sorted(df['Furnished_Status'].unique())]
furn_labels = sorted(df['Furnished_Status'].unique())
fig, ax     = plt.subplots(figsize=(10, 5))
ax.boxplot(furn_groups, labels=furn_labels, patch_artist=True)
ax.set_ylabel('Price (Lakhs)')
ax.set_title('Price Distribution by Furnished Status')
plt.tight_layout()
plt.show()""",

        "Q15": """\
import matplotlib.pyplot as plt

facing_avg = df.groupby('Facing')['Price_per_SqFt'].mean().sort_values(ascending=False)
fig, ax    = plt.subplots(figsize=(8, 5))
ax.bar(facing_avg.index, facing_avg.values, color=['#2563eb','#16a34a','#d97706','#7c3aed'],
       edgecolor='white', width=0.5)
ax.set_ylabel('Avg Price per SqFt (₹)')
ax.set_title('Price per SqFt by Facing Direction')
plt.tight_layout()
plt.show()""",

        "Q16": """\
import matplotlib.pyplot as plt

owner_counts = df['Owner_Type'].value_counts()
fig, ax      = plt.subplots(figsize=(8, 5))
ax.bar(owner_counts.index, owner_counts.values,
       color=['#2563eb','#16a34a','#7c3aed'], edgecolor='white', width=0.5)
ax.set_ylabel('Number of Properties')
ax.set_title('Properties by Owner Type')
plt.tight_layout()
plt.show()""",

        "Q17": """\
import matplotlib.pyplot as plt

avail_counts = df['Availability_Status'].value_counts()
fig, ax      = plt.subplots(figsize=(6, 5))
ax.pie(avail_counts.values, labels=avail_counts.index,
       autopct='%1.1f%%', colors=['#16a34a','#d97706'],
       wedgeprops={'edgecolor':'white','linewidth':2}, startangle=90)
ax.set_title('Properties by Availability Status')
plt.tight_layout()
plt.show()""",

        "Q18": """\
import matplotlib.pyplot as plt

park_groups = [df[df['Has_Parking']==0]['Price_in_Lakhs'].dropna(),
               df[df['Has_Parking']==1]['Price_in_Lakhs'].dropna()]
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(park_groups, labels=['No Parking','Has Parking'], patch_artist=True)
ax.set_ylabel('Price (Lakhs)')
ax.set_title('Parking Space vs Property Price')
plt.tight_layout()
plt.show()""",

        "Q19": """\
import matplotlib.pyplot as plt

amenity_avg = df.groupby('Amenity_Count')['Price_per_SqFt'].mean()
amenity_cnt = df.groupby('Amenity_Count').size()
fig, ax1    = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.bar(amenity_avg.index, amenity_avg.values, color='#2563eb',
        edgecolor='white', alpha=0.75, width=0.4, label='Avg Price/SqFt')
ax2.plot(amenity_cnt.index, amenity_cnt.values, color='red',
         marker='o', linewidth=2, label='Property Count')
ax1.set_xlabel('Amenity Count')
ax1.set_ylabel('Avg Price per SqFt (₹)')
ax2.set_ylabel('Number of Properties')
ax1.set_title('Amenities vs Price per SqFt')
plt.tight_layout()
plt.show()""",

        "Q20": """\
import matplotlib.pyplot as plt

trans_ppsf = df.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].mean()
trans_inv  = df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean() * 100
fig, axes  = plt.subplots(1, 2, figsize=(14, 4))
axes[0].bar(trans_ppsf.index, trans_ppsf.values, color='#2563eb', edgecolor='white', width=0.4)
axes[0].set_title('Transport Accessibility vs Price/SqFt')
axes[0].set_ylabel('Avg Price per SqFt (₹)')
axes[1].bar(trans_inv.index, trans_inv.values,
            color=['#16a34a' if v>=50 else '#dc2626' for v in trans_inv.values],
            edgecolor='white', width=0.4)
axes[1].axhline(50, color='gray', linestyle='--', lw=1.5)
axes[1].set_title('Transport Accessibility vs Investment Potential')
axes[1].set_ylabel('Good Investment %')
plt.tight_layout()
plt.show()""",
    }

    # ── Dropdown 1 — Group ────────────────────────────────────
    dd1, dd2 = st.columns([1, 2])
    with dd1:
        selected_group = st.selectbox(
            "📂 Step 1 — Select Group",
            list(EDA_GROUPS.keys())
        )
    with dd2:
        selected_q = st.selectbox(
            "❓ Step 2 — Select Question",
            EDA_GROUPS[selected_group]
        )

    # Extract Q number
    q_num = selected_q.split(".")[0].strip()   # e.g. "Q1"

    st.markdown("<hr class='cdivider'>", unsafe_allow_html=True)

    # ── Question banner ───────────────────────────────────────
    st.markdown(f"""
    <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                padding:14px 20px;margin-bottom:18px;'>
        <span style='font-size:1.05rem;font-weight:700;color:#1e40af;'>{selected_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Code section ──────────────────────────────────────────
    st.markdown("<div class='sec-head'>📝 Python Code</div>", unsafe_allow_html=True)
    st.code(EDA_CODE[q_num], language="python")

    # ── Chart section ─────────────────────────────────────────
    st.markdown("<div class='sec-head'>📊 Visualization</div>", unsafe_allow_html=True)

    light_style()

    def finish(fig):
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Q1 ────────────────────────────────────────────────────
    if q_num == "Q1":
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(eda['Price_in_Lakhs'], bins=60, color=BLUE, edgecolor='white', alpha=0.85)
        ax.axvline(eda['Price_in_Lakhs'].mean(),   color=RED,   linestyle='--', lw=2,
                   label=f"Mean ₹{eda['Price_in_Lakhs'].mean():.0f}L")
        ax.axvline(eda['Price_in_Lakhs'].median(), color=AMBER, linestyle='--', lw=2,
                   label=f"Median ₹{eda['Price_in_Lakhs'].median():.0f}L")
        ax.set_xlabel('Price (Lakhs)'); ax.set_ylabel('Count')
        ax.set_title('Q1 — Distribution of Property Prices')
        ax.legend(); ax.grid(True, alpha=0.35)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q2 ────────────────────────────────────────────────────
    elif q_num == "Q2":
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(eda['Size_in_SqFt'], bins=60, color=GREEN, edgecolor='white', alpha=0.85)
        ax.axvline(eda['Size_in_SqFt'].mean(),   color=RED,   linestyle='--', lw=2,
                   label=f"Mean {eda['Size_in_SqFt'].mean():.0f} sqft")
        ax.axvline(eda['Size_in_SqFt'].median(), color=AMBER, linestyle='--', lw=2,
                   label=f"Median {eda['Size_in_SqFt'].median():.0f} sqft")
        ax.set_xlabel('Size (SqFt)'); ax.set_ylabel('Count')
        ax.set_title('Q2 — Distribution of Property Sizes')
        ax.legend(); ax.grid(True, alpha=0.35)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q3 ────────────────────────────────────────────────────
    elif q_num == "Q3":
        pt_groups = [eda[eda['PType_Label']==p]['Price_per_SqFt'].dropna()
                     for p in sorted(eda['PType_Label'].unique())]
        pt_labels = sorted(eda['PType_Label'].unique())
        fig, ax   = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(pt_groups, labels=pt_labels, patch_artist=True,
                        medianprops=dict(color=AMBER, linewidth=2.5),
                        whiskerprops=dict(color='#94a3b8'), capprops=dict(color='#94a3b8'),
                        flierprops=dict(marker='o', alpha=0.25, markersize=3))
        for patch, c in zip(bp['boxes'], [BLUE, GREEN, PURPLE]):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        ax.set_ylabel('Price per SqFt (₹)')
        ax.set_title('Q3 — Price per SqFt by Property Type')
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q4 ────────────────────────────────────────────────────
    elif q_num == "Q4":
        sample = eda.sample(min(5000, len(eda)), random_state=42)
        fig, ax = plt.subplots(figsize=(12, 5))
        sc = ax.scatter(sample['Size_in_SqFt'], sample['Price_in_Lakhs'],
                        alpha=0.25, s=10, c=sample['BHK'], cmap='coolwarm')
        plt.colorbar(sc, ax=ax, label='BHK')
        m, b = np.polyfit(eda['Size_in_SqFt'], eda['Price_in_Lakhs'], 1)
        xs   = np.linspace(eda['Size_in_SqFt'].min(), eda['Size_in_SqFt'].max(), 100)
        ax.plot(xs, m*xs+b, color=RED, linewidth=2, label='Trend Line')
        ax.set_xlabel('Size (SqFt)'); ax.set_ylabel('Price (Lakhs)')
        ax.set_title('Q4 — Property Size vs Price (coloured by BHK)')
        ax.legend(); ax.grid(True, alpha=0.35)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q5 ────────────────────────────────────────────────────
    elif q_num == "Q5":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_, col, color in zip(axes, ['Price_per_SqFt','Size_in_SqFt'], [BLUE, GREEN]):
            bp = ax_.boxplot(eda[col].dropna(), vert=False, patch_artist=True,
                             medianprops=dict(color=AMBER, linewidth=2.5),
                             whiskerprops=dict(color='#94a3b8'), capprops=dict(color='#94a3b8'),
                             flierprops=dict(marker='o', alpha=0.2, markersize=4, color=RED))
            bp['boxes'][0].set_facecolor(color); bp['boxes'][0].set_alpha(0.65)
            q1_ = eda[col].quantile(0.25); q3_ = eda[col].quantile(0.75)
            out_ = eda[(eda[col] < q1_-1.5*(q3_-q1_)) | (eda[col] > q3_+1.5*(q3_-q1_))]
            ax_.set_xlabel(col)
            ax_.set_title(f'{col}\n{len(out_):,} outliers ({len(out_)/len(eda)*100:.1f}%)')
            ax_.grid(True, alpha=0.35, axis='x')
            ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
        finish(fig)

    # ── Q6 ────────────────────────────────────────────────────
    elif q_num == "Q6":
        state_avg = eda.groupby('State_Label')['Price_per_SqFt'].mean().sort_values(ascending=True)
        fig, ax   = plt.subplots(figsize=(14, 6))
        colors_   = [BLUE if v >= state_avg.median() else '#93c5fd' for v in state_avg.values]
        ax.barh(state_avg.index, state_avg.values, color=colors_, edgecolor='white', height=0.65)
        ax.axvline(state_avg.median(), color=AMBER, linestyle='--', lw=2,
                   label=f'Median: ₹{state_avg.median():.0f}')
        ax.set_xlabel('Avg Price per SqFt (₹)')
        ax.set_title('Q6 — Average Price per SqFt by State')
        ax.legend(); ax.grid(True, alpha=0.35, axis='x')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q7 ────────────────────────────────────────────────────
    elif q_num == "Q7":
        city_avg = eda.groupby('City_Label')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15)
        fig, ax  = plt.subplots(figsize=(14, 5))
        colors_  = [BLUE if i == 0 else '#93c5fd' for i in range(len(city_avg))]
        ax.bar(city_avg.index, city_avg.values, color=colors_, edgecolor='white', width=0.65)
        ax.set_xlabel('City'); ax.set_ylabel('Avg Price (Lakhs)')
        ax.set_title('Q7 — Average Property Price by City (Top 15)')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q8 ────────────────────────────────────────────────────
    elif q_num == "Q8":
        loc_age = eda.groupby('Locality_Label')['Age_of_Property'].median().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        colors_ = [RED if v >= loc_age.median() else AMBER for v in loc_age.values]
        ax.barh(loc_age.index, loc_age.values, color=colors_, edgecolor='white', height=0.65)
        ax.set_xlabel('Median Age (Years)')
        ax.set_title('Q8 — Median Age of Properties by Locality (Top 20)')
        ax.grid(True, alpha=0.35, axis='x')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q9 ────────────────────────────────────────────────────
    elif q_num == "Q9":
        top10_cities = eda['City_Label'].value_counts().head(10).index
        bhk_city     = (eda[eda['City_Label'].isin(top10_cities)]
                        .groupby(['City_Label','BHK']).size().unstack(fill_value=0))
        fig, ax = plt.subplots(figsize=(14, 5))
        bhk_city.plot(kind='bar', ax=ax, color=PAL5, edgecolor='white', width=0.75)
        ax.set_xlabel('City'); ax.set_ylabel('Count')
        ax.set_title('Q9 — BHK Distribution across Top 10 Cities')
        ax.legend(title='BHK', fontsize=9, title_fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q10 ───────────────────────────────────────────────────
    elif q_num == "Q10":
        top5_loc = (eda.groupby('Locality_Label')['Price_in_Lakhs']
                    .mean().sort_values(ascending=False).head(5).index)
        fig, ax  = plt.subplots(figsize=(14, 5))
        for loc, color in zip(top5_loc, PAL5):
            sub = eda[eda['Locality_Label']==loc].groupby('Year_Built')['Price_in_Lakhs'].mean()
            ax.plot(sub.index, sub.values, marker='o', linewidth=2, markersize=5,
                    label=loc, color=color)
        ax.set_xlabel('Year Built'); ax.set_ylabel('Avg Price (Lakhs)')
        ax.set_title('Q10 — Price Trends for Top 5 Expensive Localities')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.35)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q11 ───────────────────────────────────────────────────
    elif q_num == "Q11":
        num_cols = ['BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt',
                    'Age_of_Property','Floor_No','Total_Floors',
                    'Nearby_Schools','Nearby_Hospitals','Amenity_Count',
                    'Floor_Ratio','Good_Investment','Future_Price_5Y']
        corr = eda[num_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#ffffff')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    ax=ax, linewidths=0.5, annot_kws={'size':8,'color':'#1a1a2e'},
                    cbar_kws={'shrink':0.7})
        ax.tick_params(colors='#475569', labelsize=9)
        ax.set_title('Q11 — Numeric Feature Correlation Matrix')
        finish(fig)

    # ── Q12 ───────────────────────────────────────────────────
    elif q_num == "Q12":
        school_avg = eda.groupby('Nearby_Schools')['Price_per_SqFt'].mean()
        fig, ax    = plt.subplots(figsize=(12, 5))
        ax.bar(school_avg.index, school_avg.values, color=BLUE, edgecolor='white', alpha=0.85, width=0.6)
        ax.set_xlabel('Number of Nearby Schools')
        ax.set_ylabel('Avg Price per SqFt (₹)')
        ax.set_title('Q12 — Nearby Schools vs Price per SqFt')
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q13 ───────────────────────────────────────────────────
    elif q_num == "Q13":
        hosp_avg = eda.groupby('Nearby_Hospitals')['Price_per_SqFt'].mean()
        fig, ax  = plt.subplots(figsize=(12, 5))
        ax.bar(hosp_avg.index, hosp_avg.values, color=GREEN, edgecolor='white', alpha=0.85, width=0.6)
        ax.set_xlabel('Number of Nearby Hospitals')
        ax.set_ylabel('Avg Price per SqFt (₹)')
        ax.set_title('Q13 — Nearby Hospitals vs Price per SqFt')
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q14 ───────────────────────────────────────────────────
    elif q_num == "Q14":
        furn_groups = [eda[eda['Furnished_Label']==f]['Price_in_Lakhs'].dropna()
                       for f in sorted(eda['Furnished_Label'].unique())]
        furn_labels = sorted(eda['Furnished_Label'].unique())
        fig, ax     = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(furn_groups, labels=furn_labels, patch_artist=True,
                        medianprops=dict(color=AMBER, linewidth=2.5),
                        whiskerprops=dict(color='#94a3b8'), capprops=dict(color='#94a3b8'),
                        flierprops=dict(marker='o', alpha=0.2, markersize=3))
        for patch, c in zip(bp['boxes'], [BLUE, GREEN, PURPLE]):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        ax.set_ylabel('Price (Lakhs)')
        ax.set_title('Q14 — Price Distribution by Furnished Status')
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q15 ───────────────────────────────────────────────────
    elif q_num == "Q15":
        facing_avg = eda.groupby('Facing_Label')['Price_per_SqFt'].mean().sort_values(ascending=False)
        fig, ax    = plt.subplots(figsize=(10, 5))
        colors_    = [BLUE, GREEN, AMBER, PURPLE][:len(facing_avg)]
        bars = ax.bar(facing_avg.index, facing_avg.values, color=colors_, edgecolor='white', alpha=0.88, width=0.5)
        for bar, val in zip(bars, facing_avg.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'₹{val:.0f}', ha='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        ax.set_ylabel('Avg Price per SqFt (₹)')
        ax.set_title('Q15 — Price per SqFt by Facing Direction')
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q16 ───────────────────────────────────────────────────
    elif q_num == "Q16":
        owner_counts = eda['Owner_Label'].value_counts()
        fig, ax      = plt.subplots(figsize=(9, 5))
        bars = ax.bar(owner_counts.index, owner_counts.values,
                      color=[BLUE, GREEN, PURPLE], edgecolor='white', alpha=0.88, width=0.5)
        for bar, val in zip(bars, owner_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f'{val:,}', ha='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        ax.set_ylabel('Number of Properties')
        ax.set_title('Q16 — Properties by Owner Type')
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q17 ───────────────────────────────────────────────────
    elif q_num == "Q17":
        avail_counts = eda['Avail_Label'].value_counts()
        fig, ax      = plt.subplots(figsize=(7, 5))
        ax.pie(avail_counts.values, labels=avail_counts.index,
               autopct='%1.1f%%', colors=[GREEN, AMBER],
               textprops={'color':'#1a1a2e','fontsize':12},
               wedgeprops={'edgecolor':'white','linewidth':2}, startangle=90)
        ax.set_title('Q17 — Properties by Availability Status')
        finish(fig)

    # ── Q18 ───────────────────────────────────────────────────
    elif q_num == "Q18":
        park_groups = [eda[eda['Has_Parking']==0]['Price_in_Lakhs'].dropna(),
                       eda[eda['Has_Parking']==1]['Price_in_Lakhs'].dropna()]
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(park_groups, labels=['No Parking','Has Parking'], patch_artist=True,
                        medianprops=dict(color=AMBER, linewidth=2.5),
                        whiskerprops=dict(color='#94a3b8'), capprops=dict(color='#94a3b8'),
                        flierprops=dict(marker='o', alpha=0.2, markersize=3))
        bp['boxes'][0].set_facecolor(RED);   bp['boxes'][0].set_alpha(0.65)
        bp['boxes'][1].set_facecolor(GREEN);  bp['boxes'][1].set_alpha(0.65)
        for i, grp in enumerate(park_groups, 1):
            ax.text(i, grp.mean(), f' ₹{grp.mean():.0f}L', va='bottom',
                    fontsize=9, color='#1a1a2e', fontweight='bold')
        ax.set_ylabel('Price (Lakhs)')
        ax.set_title('Q18 — Parking Space vs Property Price')
        ax.grid(True, alpha=0.35, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        finish(fig)

    # ── Q19 ───────────────────────────────────────────────────
    elif q_num == "Q19":
        amenity_avg = eda.groupby('Amenity_Count')['Price_per_SqFt'].mean()
        amenity_cnt = eda.groupby('Amenity_Count').size()
        fig, ax1    = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.bar(amenity_avg.index, amenity_avg.values, color=BLUE,
                edgecolor='white', alpha=0.75, width=0.4, label='Avg Price/SqFt')
        ax2.plot(amenity_cnt.index, amenity_cnt.values, color=RED,
                 marker='o', linewidth=2, markersize=6, label='Property Count')
        ax1.set_xlabel('Amenity Count')
        ax1.set_ylabel('Avg Price per SqFt (₹)', color=BLUE)
        ax2.set_ylabel('Number of Properties', color=RED)
        ax1.set_title('Q19 — Amenities vs Price per SqFt')
        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, lbl1+lbl2, fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.35, axis='y')
        ax1.spines['top'].set_visible(False)
        finish(fig)

    # ── Q20 ───────────────────────────────────────────────────
    elif q_num == "Q20":
        trans_ppsf = eda.groupby('Transport_Label')['Price_per_SqFt'].mean().sort_values(ascending=False)
        trans_inv  = eda.groupby('Transport_Label')['Good_Investment'].mean().sort_values(ascending=False)*100
        fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
        colors_a = [GREEN if v == trans_ppsf.max() else BLUE for v in trans_ppsf.values]
        axes[0].bar(trans_ppsf.index, trans_ppsf.values, color=colors_a, edgecolor='white', alpha=0.88, width=0.4)
        for bar, val in zip(axes[0].patches, trans_ppsf.values):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                         f'₹{val:.0f}', ha='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        axes[0].set_ylabel('Avg Price per SqFt (₹)')
        axes[0].set_title('Transport Accessibility vs Price/SqFt')
        axes[0].grid(True, alpha=0.35, axis='y')
        axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
        colors_b = [GREEN if v >= 50 else RED for v in trans_inv.values]
        axes[1].bar(trans_inv.index, trans_inv.values, color=colors_b, edgecolor='white', alpha=0.88, width=0.4)
        axes[1].axhline(50, color='#94a3b8', linestyle='--', lw=1.5)
        for bar, val in zip(axes[1].patches, trans_inv.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        axes[1].set_ylabel('Good Investment %')
        axes[1].set_title('Transport Accessibility vs Investment Potential')
        axes[1].grid(True, alpha=0.35, axis='y')
        axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
        finish(fig)


# ==============================================================
#  PAGE 5 — MODEL PERFORMANCE
# ==============================================================
elif page == "🤖 Model Performance":
    st.markdown("""
    <h1 style='color:#1a1a2e;font-size:2rem;font-weight:800;'>🤖 Model Performance</h1>
    <p style='color:#64748b;'>Compare all 10 trained models — 5 classifiers + 5 regressors.</p>
    <hr class='cdivider'>
    """, unsafe_allow_html=True)

    b1,b2 = st.columns(2)
    with b1:
        st.markdown(f"""
        <div class='best-clf'>
            <div style='font-size:1.6rem;'>🏆</div>
            <div style='color:#64748b;font-size:0.8rem;margin:4px 0;'>BEST CLASSIFIER</div>
            <div style='color:#16a34a;font-size:1.3rem;font-weight:800;'>{model_info['best_classifier_name']}</div>
            <div style='color:#374151;font-size:0.85rem;margin-top:8px;'>
                Accuracy: <b>{model_info['clf_metrics']['Accuracy']}</b> &nbsp;|&nbsp;
                ROC-AUC: <b>{model_info['clf_metrics']['ROC_AUC']}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    with b2:
        st.markdown(f"""
        <div class='best-reg'>
            <div style='font-size:1.6rem;'>🏆</div>
            <div style='color:#64748b;font-size:0.8rem;margin:4px 0;'>BEST REGRESSOR</div>
            <div style='color:#2563eb;font-size:1.3rem;font-weight:800;'>{model_info['best_regressor_name']}</div>
            <div style='color:#374151;font-size:0.85rem;margin-top:8px;'>
                RMSE: <b>{model_info['reg_metrics']['RMSE']}</b> &nbsp;|&nbsp;
                R²: <b>{model_info['reg_metrics']['R2']}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    light_style()

    # Classification
    st.markdown("<div class='sec-head'>🔵 Classification Results</div>", unsafe_allow_html=True)
    clf_data = pd.DataFrame(model_info['clf_all_results'])
    st.dataframe(clf_data.style
        .highlight_max(subset=['Accuracy','Precision','Recall','F1_Score','ROC_AUC'], color='#dcfce7')
        .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}',
                 'F1_Score':'{:.4f}','ROC_AUC':'{:.4f}'}),
        use_container_width=True)

    metrics_clf = ['Accuracy','Precision','Recall','F1_Score','ROC_AUC']
    x, w = np.arange(len(clf_data)), 0.15
    fig,ax = plt.subplots(figsize=(14,5))
    for i,(m,c) in enumerate(zip(metrics_clf, PAL5)):
        ax.bar(x+i*w, clf_data[m], w, label=m, color=c, edgecolor='white', alpha=0.88)
    ax.set_xticks(x+w*2)
    ax.set_xticklabels(clf_data['Model'], rotation=15, ha='right')
    ax.set_ylim(0,1.15); ax.set_ylabel('Score')
    ax.set_title('Classification Models — All Metrics Comparison')
    ax.legend(fontsize=9)
    ax.grid(True,alpha=0.4,axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Regression
    st.markdown("<div class='sec-head'>🟠 Regression Results</div>", unsafe_allow_html=True)
    reg_data = pd.DataFrame(model_info['reg_all_results'])
    st.dataframe(reg_data.style
        .highlight_max(subset=['R2'], color='#dcfce7')
        .highlight_min(subset=['RMSE','MAE'], color='#dcfce7')
        .format({'RMSE':'{:.4f}','MAE':'{:.4f}','R2':'{:.4f}'}),
        use_container_width=True)

    fig,axes = plt.subplots(1,3,figsize=(16,5))
    for ax_,(metric,color,note) in zip(axes,[
        ('RMSE',RED,  'Lower is Better ↓'),
        ('MAE', AMBER,'Lower is Better ↓'),
        ('R2',  GREEN,'Higher is Better ↑')
    ]):
        bars = ax_.bar(reg_data['Model'], reg_data[metric],
                       color=color, edgecolor='white', alpha=0.88, width=0.5)
        for bar,val in zip(bars, reg_data[metric]):
            ax_.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                     f'{val:.4f}', ha='center', fontsize=8.5, color='#1a1a2e', fontweight='600')
        ax_.set_title(f'{metric} — {note}')
        ax_.tick_params(axis='x', rotation=20)
        ax_.grid(True,alpha=0.4,axis='y')
        ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ==============================================================
#  PAGE 6 — ABOUT CREATOR
# ==============================================================
elif page == "👩‍💻 About Creator":
    st.markdown("""
    <h1 style='color:#1a1a2e;font-size:2rem;font-weight:800;'>👩‍💻 About Creator</h1>
    <p style='color:#64748b;'>The person behind this project.</p>
    <hr class='cdivider'>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2.5], gap="large")

    with col1:
        photo_path = 'kavya_photo.jpeg'
        if os.path.exists(photo_path):
            # Small, circular-style photo using CSS
            from PIL import Image
            img = Image.open(photo_path)
            # Crop to square from center
            w_, h_ = img.size
            side   = min(w_, h_)
            left   = (w_ - side) // 2
            top    = (h_ - side) // 2
            img    = img.crop((left, top, left+side, top+side))
            img    = img.resize((200, 200))
            st.image(img, width=180)
        else:
            st.markdown("""
            <div style='width:150px;height:150px;border-radius:50%;
                        background:linear-gradient(135deg,#2563eb,#7c3aed);
                        display:flex;align-items:center;justify-content:center;
                        font-size:4rem;'>👩‍💻</div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:14px;'>
            <div style='font-size:1.3rem;font-weight:800;color:#1a1a2e;'>KAVYA S</div>
            <div style='font-size:0.9rem;color:#2563eb;font-weight:600;margin:3px 0;'>
                Data Science Student
            </div>
            <div style='font-size:0.8rem;color:#64748b;'>
                B.E Biomedical Engineering<br>Minor in AI & Data Science
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='sec-head'>👋 About Me</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='color:#374151;line-height:1.75;'>
            I am <b style='color:#2563eb;'>Kavya S</b>, a passionate Data Science student
            pursuing B.E in Biomedical Engineering with a Minor in
            <b style='color:#7c3aed;'>Artificial Intelligence & Data Science</b>.<br><br>
            I built this <b>Real Estate Investment Advisor</b> to combine my love for
            data science with real-world problem solving — helping investors make smarter,
            data-backed property decisions using Machine Learning.
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-head'>🎓 Education</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.04);'>
            <div style='display:flex;align-items:center;gap:10px;'>
                <span style='font-size:1.4rem;'>🎓</span>
                <div>
                    <div style='color:#1a1a2e;font-weight:700;'>B.E Biomedical Engineering</div>
                    <div style='color:#2563eb;font-size:0.86rem;'>Minor in Artificial Intelligence & Data Science</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-head'>💼 Technical Skills</div>", unsafe_allow_html=True)
        skills = ["Python","Pandas","NumPy","Scikit-learn","XGBoost",
                  "Machine Learning","Deep Learning","Data Visualization",
                  "Streamlit","MLflow","Matplotlib","Seaborn","SQL","EDA"]
        st.markdown(
            "<div style='line-height:2.3;'>" +
            "".join([f"<span class='skill-badge'>{s}</span>" for s in skills]) +
            "</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-head'>📌 About This Project</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='color:#374151;line-height:1.75;'>
            <b style='color:#d97706;'>Real Estate Investment Advisor</b> is an end-to-end ML
            project built on <b>2,50,000 Indian housing records</b>. It performs:<br><br>
            • <b style='color:#16a34a;'>Classification</b> — Predict if a property is a Good Investment<br>
            • <b style='color:#2563eb;'>Regression</b> — Predict property price after 5 years<br><br>
            Built using <b>XGBoost, Random Forest, Scikit-learn, MLflow</b> and deployed with <b>Streamlit</b>.
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-head'>📞 Contact</div>", unsafe_allow_html=True)
        for icon,label,value,link in [
            ("📧","Email",    "kavya22s145@gmail.com",          "mailto:kavya22s145@gmail.com"),
            ("💼","LinkedIn", "linkedin.com/in/kavya-s1245",    "https://www.linkedin.com/in/kavya-s1245/"),
            ("🐙","GitHub",   "github.com/Kavya1245",           "https://github.com/Kavya1245"),
        ]:
            st.markdown(f"""
            <div class='contact-card'>
                <div style='display:flex;align-items:center;gap:12px;'>
                    <span style='font-size:1.2rem;'>{icon}</span>
                    <div>
                        <div style='color:#94a3b8;font-size:0.73rem;'>{label}</div>
                        <a href='{link}' target='_blank'
                           style='color:#2563eb;font-weight:600;font-size:0.88rem;text-decoration:none;'>
                           {value}</a>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        Built with ❤️ by <b style='color:#2563eb;'>Kavya S</b> &nbsp;·&nbsp;
        Real Estate Investment Advisor &nbsp;·&nbsp;
        Powered by XGBoost + Streamlit + MLflow
    </div>""", unsafe_allow_html=True)
