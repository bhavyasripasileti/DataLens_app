import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from io import StringIO, BytesIO
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="DataLens · CSV Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e8eaf0; }

section[data-testid="stSidebar"] { background: #13161e; border-right: 1px solid #1f2330; }
section[data-testid="stSidebar"] * { color: #c9ccd6 !important; }

h1 { font-family: 'Space Mono', monospace !important; color: #a8f0c6 !important; letter-spacing: -1px; }
h2 { font-family: 'Space Mono', monospace !important; color: #7eb8f7 !important; font-size: 1.1rem !important; }
h3 { font-family: 'DM Sans', sans-serif !important; color: #c9ccd6 !important; }

[data-testid="metric-container"] {
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 12px; padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #7eb8f7 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #a8f0c6 !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }

[data-baseweb="tab-list"] { background: #13161e !important; border-radius: 10px; gap: 4px; padding: 4px; }
[data-baseweb="tab"] { border-radius: 8px !important; color: #7a7f94 !important; font-size: 0.85rem !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #1f2330 !important; color: #a8f0c6 !important; }

[data-testid="stDataFrame"] { border: 1px solid #1f2330; border-radius: 10px; overflow: hidden; }
[data-testid="stFileUploaderDropzone"] {
    background: #13161e !important; border: 2px dashed #2a3050 !important;
    border-radius: 14px !important; padding: 32px !important;
}
[data-testid="stExpander"] { background: #13161e; border: 1px solid #1f2330; border-radius: 10px; }
[data-baseweb="select"] { background: #13161e !important; }
hr { border-color: #1f2330 !important; }

.ml-card {
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 14px; padding: 20px 24px; margin-bottom: 14px;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a3050; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# Matplotlib / Seaborn theme
plt.rcParams.update({
    "figure.facecolor": "#13161e", "axes.facecolor": "#13161e",
    "axes.edgecolor": "#1f2330",   "axes.labelcolor": "#c9ccd6",
    "xtick.color": "#7a7f94",      "ytick.color": "#7a7f94",
    "text.color": "#c9ccd6",       "grid.color": "#1f2330",
    "grid.linestyle": "--",        "grid.alpha": 0.6,
    "font.family": "monospace",    "axes.titlesize": 11, "axes.labelsize": 9,
})
ACCENT  = "#a8f0c6"
ACCENT2 = "#7eb8f7"
ACCENT3 = "#f0a8d0"
PAL     = [ACCENT, ACCENT2, ACCENT3, "#f7c97e", "#b8a8f0", "#f07e7e"]

# Helpers 
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

def missing_summary(df):
    miss = df.isnull().sum()
    pct  = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"Missing Count": miss, "Missing %": pct, "Dtype": df.dtypes})[miss > 0].sort_values("Missing %", ascending=False)

# Auto ML logic
MODEL_MAP = {
    "binary_classification": {
        "emoji": "🎯", "label": "Binary Classification", "color": "#a8f0c6",
        "models": [
            ("Logistic Regression",    "Great baseline, highly interpretable"),
            ("Random Forest",          "Handles non-linearity, robust to outliers"),
            ("XGBoost / LightGBM",     "Top performance on tabular data"),
            ("Support Vector Machine", "Effective in high-dimensional spaces"),
            ("Neural Network (MLP)",   "Good for large datasets with complex patterns"),
        ],
        "tips": [
            "Check class balance — use SMOTE or class_weight if imbalanced",
            "Use ROC-AUC, F1, and Precision-Recall as evaluation metrics",
            "Feature scaling recommended for Logistic Regression & SVM",
        ],
    },
    "multiclass_classification": {
        "emoji": "🏷️", "label": "Multi-class Classification", "color": "#7eb8f7",
        "models": [
            ("Random Forest",          "Naturally supports multi-class, robust"),
            ("XGBoost / LightGBM",     "Excellent performance with softmax objective"),
            ("K-Nearest Neighbors",    "Simple, effective for smaller datasets"),
            ("Neural Network (MLP)",   "Scalable to many classes"),
            ("Multinomial Naive Bayes","Fast baseline for text-like features"),
        ],
        "tips": [
            "Use macro/weighted F1 score for evaluation",
            "One-vs-Rest strategy works well for simpler models",
            "Label encoding or one-hot needed for categorical features",
        ],
    },
    "regression": {
        "emoji": "📈", "label": "Regression", "color": "#f7c97e",
        "models": [
            ("Linear Regression",          "Interpretable baseline, fast to train"),
            ("Ridge / Lasso",              "Regularized — handles multicollinearity"),
            ("Random Forest Regressor",    "Non-linear, handles outliers well"),
            ("XGBoost / LightGBM",         "State-of-the-art for tabular regression"),
            ("SVR",                        "Effective for smaller high-dimensional data"),
        ],
        "tips": [
            "Use RMSE, MAE, and R² as primary metrics",
            "Log-transform skewed targets for better model fit",
            "Check for multicollinearity before using linear models",
        ],
    },
}

def get_problem_type(df, col, num_cols):
    nuniq = df[col].nunique()
    if col in num_cols and nuniq > 20:
        return "regression"
    elif nuniq == 2:
        return "binary_classification"
    else:
        return "multiclass_classification"

def auto_detect_recommendations(df, num_cols, cat_cols):
    results, seen = [], set()
    for col in df.columns:
        ptype = get_problem_type(df, col, num_cols)
        if ptype not in seen:
            seen.add(ptype)
            rec = MODEL_MAP[ptype].copy()
            rec["candidate_cols"] = [c for c in df.columns if get_problem_type(df, c, num_cols) == ptype][:5]
            results.append(rec)
    return results

# Report generator
def generate_report(df, num_cols, cat_cols, filename):
    ms       = missing_summary(df)
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    miss_pct = f"{df.isnull().sum().sum() / df.size * 100:.1f}"

    corr_section = ""
    if len(num_cols) >= 2:
        corr  = df[num_cols].corr()
        pairs = (corr.where(np.tril(np.ones_like(corr, dtype=bool), k=-1))
                     .stack().reset_index()
                     .rename(columns={"level_0":"Col A","level_1":"Col B",0:"r"})
                     .assign(abs_r=lambda x: x["r"].abs())
                     .sort_values("abs_r", ascending=False).head(10))
        rows = "".join(
            f"<tr><td>{r['Col A']}</td><td>{r['Col B']}</td>"
            f"<td style='color:{'#a8f0c6' if r['r']>0 else '#f0a8d0'}'>{r['r']:.3f}</td></tr>"
            for _, r in pairs.iterrows()
        )
        corr_section = f"""<h2>🔗 Top Correlations</h2>
        <table><thead><tr><th>Column A</th><th>Column B</th><th>Correlation</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    if ms.empty:
        miss_section = "<p style='color:#a8f0c6'>✅ No missing values detected.</p>"
    else:
        rows2 = "".join(
            f"<tr><td>{idx}</td><td>{int(row['Missing Count'])}</td><td>{row['Missing %']}%</td><td>{row['Dtype']}</td></tr>"
            for idx, row in ms.iterrows()
        )
        miss_section = f"""<table><thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Dtype</th></tr></thead>
        <tbody>{rows2}</tbody></table>"""

    stat_rows = ""
    for col in df.columns[:20]:
        nuniq = df[col].nunique()
        nmiss = df[col].isnull().sum()
        if col in num_cols:
            mean, std = f"{df[col].mean():.2f}", f"{df[col].std():.2f}"
            mn,   mx  = f"{df[col].min():.2f}",  f"{df[col].max():.2f}"
        else:
            mean = std = mn = mx = "—"
        stat_rows += f"<tr><td>{col}</td><td>{str(df[col].dtype)}</td><td>{nuniq}</td><td>{nmiss}</td><td>{mean}</td><td>{std}</td><td>{mn}</td><td>{mx}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>DataLens Report — {filename}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#0d0f14; color:#c9ccd6; padding:40px; }}
  .header {{ border-bottom:2px solid #1f2330; padding-bottom:24px; margin-bottom:32px; }}
  .header h1 {{ font-size:2rem; color:#a8f0c6; letter-spacing:-1px; }}
  .header p  {{ color:#7a7f94; margin-top:6px; font-size:0.9rem; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; margin-bottom:36px; }}
  .kpi {{ background:#13161e; border:1px solid #1f2330; border-radius:12px; padding:18px; }}
  .kpi .label {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; color:#7eb8f7; margin-bottom:6px; }}
  .kpi .value {{ font-size:1.6rem; font-family:monospace; color:#a8f0c6; font-weight:700; }}
  h2 {{ font-size:1rem; color:#7eb8f7; font-family:monospace; text-transform:uppercase;
        letter-spacing:1px; margin:32px 0 14px; border-left:3px solid #7eb8f7; padding-left:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  th {{ background:#1f2330; color:#7eb8f7; text-align:left; padding:10px 14px; font-weight:600; }}
  td {{ padding:8px 14px; border-bottom:1px solid #1a1d28; color:#c9ccd6; }}
  tr:hover td {{ background:#13161e; }}
  .footer {{ margin-top:48px; padding-top:20px; border-top:1px solid #1f2330;
             font-size:0.75rem; color:#7a7f94; text-align:center; }}
</style></head><body>
<div class='header'>
  <h1>🔬 DataLens Report</h1>
  <p>File: <strong style='color:#c9ccd6'>{filename}</strong> &nbsp;·&nbsp; Generated: {now}</p>
</div>
<div class='kpi-grid'>
  <div class='kpi'><div class='label'>Rows</div><div class='value'>{df.shape[0]:,}</div></div>
  <div class='kpi'><div class='label'>Columns</div><div class='value'>{df.shape[1]:,}</div></div>
  <div class='kpi'><div class='label'>Numeric</div><div class='value'>{len(num_cols)}</div></div>
  <div class='kpi'><div class='label'>Categorical</div><div class='value'>{len(cat_cols)}</div></div>
  <div class='kpi'><div class='label'>Missing %</div><div class='value'>{miss_pct}%</div></div>
</div>
<h2>🩺 Missing Values</h2>{miss_section}
<h2>📐 Column Statistics</h2>
<table><thead><tr><th>Column</th><th>Dtype</th><th>Unique</th><th>Missing</th>
<th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>
<tbody>{stat_rows}</tbody></table>
{corr_section}
<div class='footer'>Generated by DataLens · {now}</div>
</body></html>"""
    return html.encode("utf-8")

# Sidebar 
with st.sidebar:
    st.markdown("## 🔬 DataLens")
    st.markdown("<span style='color:#7a7f94;font-size:0.78rem;'>CSV Explorer & Visualizer</span>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    st.markdown("---")
    st.markdown("<span style='color:#7a7f94;font-size:0.75rem;'>Supports UTF-8 encoded CSVs up to 200 MB</span>", unsafe_allow_html=True)

# Main 
st.title("DataLens")
st.markdown("<p style='color:#7a7f94;margin-top:-12px;'>Upload a CSV to explore, diagnose, and visualise your dataset instantly.</p>", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:80px 0;'>
        <div style='font-size:3.5rem;'>📂</div>
        <p style='color:#7a7f94;margin-top:12px;'>Use the sidebar to upload a CSV file and get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df       = load_data(uploaded)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# KPI row 
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",           f"{df.shape[0]:,}")
c2.metric("Columns",        f"{df.shape[1]:,}")
c3.metric("Numeric cols",   len(num_cols))
c4.metric("Categoric cols", len(cat_cols))
miss_total = df.isnull().sum().sum()
c5.metric("Missing cells",  f"{miss_total:,}", delta=f"{miss_total/df.size*100:.1f}% of total", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋  Preview", "🩺  Missing Values", "🔗  Correlation",
    "📊  Visualisations", "🤖  Auto ML", "📥  Report"
])

# ─── Tab 1 · Preview 
with tab1:
    st.markdown("## Dataset Preview")
    n_rows = st.slider("Rows to display", 5, min(200, len(df)), 10, key="preview_rows")
    st.dataframe(df.head(n_rows), use_container_width=True, height=380)
    with st.expander("📐 Shape & Column Info"):
        buf = StringIO(); df.info(buf=buf); st.code(buf.getvalue(), language="")
    with st.expander("📈 Descriptive Statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

# ─── Tab 2 · Missing Values 
with tab2:
    st.markdown("## Missing Value Analysis")
    ms = missing_summary(df)
    if ms.empty:
        st.success("✅ No missing values detected in this dataset.")
    else:
        colA, colB = st.columns([1, 1.6])
        with colA:
            st.markdown("### Summary table")
            st.dataframe(ms.style.background_gradient(subset=["Missing %"], cmap="YlOrRd"), use_container_width=True)
        with colB:
            st.markdown("### Missing % per column")
            fig, ax = plt.subplots(figsize=(6, max(3, len(ms)*0.45)))
            bars = ax.barh(ms.index, ms["Missing %"], color=ACCENT3, edgecolor="none", height=0.55)
            ax.set_xlabel("Missing %"); ax.invert_yaxis()
            ax.axvline(5, color="#f7c97e", lw=1, ls="--", alpha=0.7)
            ax.set_title("Missing values by column", pad=10)
            for bar, val in zip(bars, ms["Missing %"]):
                ax.text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=8, color="#c9ccd6")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

        st.markdown("### Missingness pattern heatmap")
        sample = df[ms.index].head(300).isnull().astype(int)
        fig2, ax2 = plt.subplots(figsize=(max(6, len(ms)*0.7), 3.5))
        sns.heatmap(sample.T, cmap=["#1f2330", ACCENT3], cbar=False,
                    linewidths=0, ax=ax2, yticklabels=True, xticklabels=False)
        ax2.set_xlabel("Row index (first 300)")
        ax2.set_title("Pink = missing  ·  Dark = present", pad=8)
        fig2.tight_layout(); st.pyplot(fig2, use_container_width=True)

# ─── Tab 3 · Correlation 
with tab3:
    st.markdown("## Correlation Heatmap")
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns to compute correlations.")
    else:
        method   = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
        sel_cols = st.multiselect("Select columns", num_cols, default=num_cols[:min(12, len(num_cols))])
        if len(sel_cols) < 2:
            st.info("Select at least 2 columns.")
        else:
            corr = df[sel_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(max(6, len(sel_cols)*0.75), max(5, len(sel_cols)*0.65)))
            cmap = sns.diverging_palette(210, 150, s=80, l=40, as_cmap=True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                        linewidths=0.5, linecolor="#0d0f14",
                        annot_kws={"size": 8}, ax=ax, vmin=-1, vmax=1,
                        cbar_kws={"shrink": 0.7, "pad": 0.02})
            ax.set_title(f"{method.capitalize()} correlation matrix", pad=12)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

            pairs = (corr.where(np.tril(np.ones_like(corr, dtype=bool), k=-1))
                         .stack().reset_index()
                         .rename(columns={"level_0":"Col A","level_1":"Col B",0:"Correlation"})
                         .assign(AbsCorr=lambda x: x["Correlation"].abs())
                         .sort_values("AbsCorr", ascending=False).drop(columns="AbsCorr"))
            with st.expander("🔍 Top correlated pairs"):
                st.dataframe(pairs.head(15).style.background_gradient(subset=["Correlation"], cmap="RdYlGn"), use_container_width=True)

# ─── Tab 4 · Visualisations 
with tab4:
    st.markdown("## Visualisations")
    vtype = st.selectbox("Chart type", ["Histogram","Box Plot","Scatter Plot","Bar Chart (categorical)","Line Chart","Pair Plot"])

    if vtype == "Histogram":
        col = st.selectbox("Numeric column", num_cols)
        bins = st.slider("Bins", 10, 100, 30)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[col].dropna(), bins=bins, color=ACCENT, edgecolor="#0d0f14", alpha=0.9)
        ax.set_xlabel(col); ax.set_ylabel("Frequency"); ax.set_title(f"Distribution of {col}")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif vtype == "Box Plot":
        cols_sel = st.multiselect("Numeric columns", num_cols, default=num_cols[:min(6, len(num_cols))])
        if cols_sel:
            fig, ax = plt.subplots(figsize=(max(6, len(cols_sel)*1.2), 5))
            bp = ax.boxplot([df[c].dropna().values for c in cols_sel], patch_artist=True, labels=cols_sel,
                            medianprops=dict(color="#0d0f14", linewidth=2))
            colors = PAL * (len(cols_sel)//len(PAL)+1)
            for patch, clr in zip(bp["boxes"], colors): patch.set_facecolor(clr); patch.set_alpha(0.85)
            ax.set_title("Box plots"); plt.xticks(rotation=30, ha="right")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif vtype == "Scatter Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            cx, cy   = st.columns(2)
            x_col    = cx.selectbox("X axis", num_cols, index=0)
            y_col    = cy.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
            hue_col  = st.selectbox("Colour by (optional)", ["None"] + cat_cols + num_cols)
            sample_n = st.slider("Max points", 100, min(5000, len(df)), min(1000, len(df)))
            plot_df  = df[[x_col, y_col] + ([hue_col] if hue_col != "None" else [])].dropna()
            subset   = plot_df.sample(min(sample_n, len(plot_df)), random_state=42)
            fig, ax  = plt.subplots(figsize=(8, 5))
            if hue_col == "None":
                ax.scatter(subset[x_col], subset[y_col], color=ACCENT, alpha=0.5, s=20, edgecolors="none")
            else:
                for i, cat in enumerate(subset[hue_col].astype(str).unique()):
                    mask = subset[hue_col].astype(str) == cat
                    ax.scatter(subset.loc[mask, x_col], subset.loc[mask, y_col],
                               color=PAL[i%len(PAL)], label=cat, alpha=0.55, s=20, edgecolors="none")
                ax.legend(fontsize=7, framealpha=0.2, loc="best")
            ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"{x_col} vs {y_col}")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif vtype == "Bar Chart (categorical)":
        if not cat_cols:
            st.warning("No categorical columns found.")
        else:
            cat_col = st.selectbox("Categorical column", cat_cols)
            top_n   = st.slider("Top N categories", 5, 30, 10)
            counts  = df[cat_col].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(range(len(counts)), counts.values, color=ACCENT2, edgecolor="none")
            ax.set_xticks(range(len(counts))); ax.set_xticklabels(counts.index, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Count"); ax.set_title(f"Top {top_n} values · {cat_col}")
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+counts.max()*0.01,
                        f"{int(bar.get_height()):,}", ha="center", fontsize=7, color="#c9ccd6")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif vtype == "Line Chart":
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            y_cols = st.multiselect("Y columns", num_cols, default=num_cols[:min(3, len(num_cols))])
            x_col  = st.selectbox("X axis", ["Row index"] + df.columns.tolist())
            if y_cols:
                fig, ax = plt.subplots(figsize=(10, 4))
                for i, col in enumerate(y_cols):
                    x_vals = df.index if x_col == "Row index" else df[x_col]
                    ax.plot(x_vals, df[col], color=PAL[i%len(PAL)], lw=1.5, label=col, alpha=0.9)
                ax.legend(fontsize=8, framealpha=0.2)
                ax.set_xlabel(x_col if x_col != "Row index" else "Index"); ax.set_title("Line chart")
                fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif vtype == "Pair Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            max_cols = st.multiselect("Columns (max 5 recommended)", num_cols, default=num_cols[:min(4, len(num_cols))])
            if len(max_cols) >= 2:
                sample_pp = df[max_cols].dropna().sample(min(500, len(df)), random_state=0)
                fig = sns.pairplot(sample_pp, plot_kws={"alpha":0.45,"s":12,"color":ACCENT},
                                   diag_kws={"color":ACCENT2,"alpha":0.8})
                fig.figure.patch.set_facecolor("#13161e")
                for ax_row in fig.axes:
                    for ax_ in ax_row:
                        ax_.set_facecolor("#13161e")
                        for spine in ax_.spines.values(): spine.set_edgecolor("#1f2330")
                st.pyplot(fig.figure, use_container_width=True)

# ─── Tab 5 · Auto ML 
with tab5:
    st.markdown("## 🤖 Auto ML Recommendation")
    st.markdown("<p style='color:#7a7f94;margin-top:-10px;'>Based on your dataset structure, here are recommended ML approaches and models.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    miss_pct = round(df.isnull().sum().sum() / df.size * 100, 1)
    with col1:
        st.markdown(f"""<div class='ml-card'>
            <div style='color:#7eb8f7;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Dataset Size</div>
            <div style='font-size:1.4rem;font-family:monospace;color:#a8f0c6;'>{df.shape[0]:,} × {df.shape[1]}</div>
            <div style='color:#7a7f94;font-size:0.8rem;margin-top:4px;'>rows × columns</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='ml-card'>
            <div style='color:#7eb8f7;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Feature Mix</div>
            <div style='font-size:1.4rem;font-family:monospace;color:#a8f0c6;'>{len(num_cols)} num · {len(cat_cols)} cat</div>
            <div style='color:#7a7f94;font-size:0.8rem;margin-top:4px;'>numeric · categorical</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        dq_color = "#f0a8d0" if miss_pct > 10 else "#a8f0c6"
        st.markdown(f"""<div class='ml-card'>
            <div style='color:#7eb8f7;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Data Quality</div>
            <div style='font-size:1.4rem;font-family:monospace;color:{dq_color};'>{miss_pct}% missing</div>
            <div style='color:#7a7f94;font-size:0.8rem;margin-top:4px;'>{'⚠️ Imputation needed' if miss_pct > 5 else '✅ Looks clean'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Select Your Target Column")
    target_col = st.selectbox("Which column do you want to predict?", ["— Auto Detect —"] + df.columns.tolist())

    if target_col == "— Auto Detect —":
        recommendations = auto_detect_recommendations(df, num_cols, cat_cols)
    else:
        ptype = get_problem_type(df, target_col, num_cols)
        rec   = MODEL_MAP[ptype].copy()
        rec["candidate_cols"] = [target_col]
        recommendations = [rec]

    for rec in recommendations:
        color  = rec["color"]
        cands  = rec.get("candidate_cols", [])
        badges = " ".join([f'<code style="background:#1f2330;padding:1px 6px;border-radius:4px;color:#c9ccd6;">{c}</code>' for c in cands[:5]])
        st.markdown(f"""<div class='ml-card' style='border-left:3px solid {color};'>
            <div style='font-size:1.1rem;font-weight:600;color:{color};margin-bottom:4px;'>{rec['emoji']} {rec['label']}</div>
            <div style='color:#7a7f94;font-size:0.8rem;'>Likely target columns: {badges}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**📦 Recommended Models**")
        for i, (model, desc) in enumerate(rec["models"]):
            rank_color = ["#f7c97e","#c9ccd6","#c9ccd6","#7a7f94","#7a7f94"][i]
            st.markdown(f"""<div style='display:flex;align-items:flex-start;padding:10px 16px;background:#13161e;
                        border:1px solid #1f2330;border-radius:10px;margin-bottom:8px;'>
                <div style='font-family:monospace;color:{rank_color};font-size:1rem;min-width:32px;'>#{i+1}</div>
                <div>
                    <div style='color:#e8eaf0;font-weight:500;'>{model}</div>
                    <div style='color:#7a7f94;font-size:0.8rem;margin-top:2px;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("**💡 Tips**")
        for tip in rec["tips"]:
            st.markdown(f"<div style='padding:6px 12px;color:#c9ccd6;font-size:0.85rem;'>→ {tip}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🔧 Preprocessing Checklist"):
        dup_count = df.duplicated().sum()
        checks = [
            ("Handle missing values",       miss_pct > 0,      "Impute or drop rows/columns with nulls"),
            ("Encode categorical features", len(cat_cols) > 0, "Use Label Encoding or One-Hot Encoding"),
            ("Scale numeric features",      len(num_cols) > 0, "StandardScaler or MinMaxScaler for distance-based models"),
            ("Remove duplicate rows",       dup_count > 0,     f"{dup_count} duplicate rows detected"),
            ("Check class imbalance",       True,              "Use SMOTE, class_weight, or oversampling if needed"),
            ("Train / Val / Test split",    True,              "Typical split: 70% train · 15% val · 15% test"),
        ]
        for label, flag, note in checks:
            icon = "⚠️" if flag else "✅"
            st.markdown(f"""<div style='display:flex;gap:10px;padding:8px 12px;border-bottom:1px solid #1a1d28;'>
                <span>{icon}</span>
                <div>
                    <div style='color:#c9ccd6;font-size:0.85rem;'>{label}</div>
                    <div style='color:#7a7f94;font-size:0.78rem;'>{note}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ─── Tab 6 · Report
with tab6:
    st.markdown("## 📥 Downloadable Report")
    st.markdown("<p style='color:#7a7f94;margin-top:-10px;'>Generate a full dark-themed HTML report of your dataset analysis.</p>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("""<div class='ml-card'>
            <div style='color:#a8f0c6;font-size:1rem;font-weight:600;margin-bottom:10px;'>📄 Report Includes</div>
            <div style='color:#c9ccd6;font-size:0.85rem;line-height:2;'>
                ✅ &nbsp; Dataset overview (rows, columns, dtypes)<br>
                ✅ &nbsp; KPI summary cards<br>
                ✅ &nbsp; Missing value analysis per column<br>
                ✅ &nbsp; Full column statistics (mean, std, min, max)<br>
                ✅ &nbsp; Top 10 correlated column pairs<br>
                ✅ &nbsp; Timestamp &amp; filename metadata
            </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class='ml-card'>
            <div style='color:#7eb8f7;font-size:1rem;font-weight:600;margin-bottom:10px;'>⚡ Format</div>
            <div style='color:#c9ccd6;font-size:0.85rem;line-height:2;'>
                📄 HTML file<br>
                🎨 Dark themed<br>
                🌐 Opens in any browser<br>
                💾 Save as PDF from browser
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⬇️ Generate & Download Report", use_container_width=True):
        report_bytes = generate_report(df, num_cols, cat_cols, uploaded.name)
        st.download_button(
            label    = "📥 Click to Download Report",
            data     = report_bytes,
            file_name= f"DataLens_Report_{uploaded.name.replace('.csv','')}.html",
            mime     = "text/html",
            use_container_width=True,
        )
        st.success("✅ Report generated! Click the button above to download.")

    st.markdown("---")
    st.markdown("**💡 Tip:** Open the downloaded HTML in your browser → `Ctrl+P` → `Save as PDF` to get a PDF version.")
