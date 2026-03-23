import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ── Page config 
st.set_page_config(
    page_title="DataLens · CSV Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1f2330;
}
section[data-testid="stSidebar"] * {
    color: #c9ccd6 !important;
}

/* Headings */
h1 { font-family: 'Space Mono', monospace !important; color: #a8f0c6 !important; letter-spacing: -1px; }
h2 { font-family: 'Space Mono', monospace !important; color: #7eb8f7 !important; font-size: 1.1rem !important; }
h3 { font-family: 'DM Sans', sans-serif !important; color: #c9ccd6 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #7eb8f7 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #a8f0c6 !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: #13161e !important; border-radius: 10px; gap: 4px; padding: 4px; }
[data-baseweb="tab"] { border-radius: 8px !important; color: #7a7f94 !important; font-size: 0.85rem !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #1f2330 !important; color: #a8f0c6 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1f2330; border-radius: 10px; overflow: hidden; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #13161e !important;
    border: 2px dashed #2a3050 !important;
    border-radius: 14px !important;
    padding: 32px !important;
}

/* Expander */
[data-testid="stExpander"] { background: #13161e; border: 1px solid #1f2330; border-radius: 10px; }

/* Selectbox / multiselect */
[data-baseweb="select"] { background: #13161e !important; }

/* Divider */
hr { border-color: #1f2330 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a3050; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib / Seaborn dark theme 
plt.rcParams.update({
    "figure.facecolor":  "#13161e",
    "axes.facecolor":    "#13161e",
    "axes.edgecolor":    "#1f2330",
    "axes.labelcolor":   "#c9ccd6",
    "xtick.color":       "#7a7f94",
    "ytick.color":       "#7a7f94",
    "text.color":        "#c9ccd6",
    "grid.color":        "#1f2330",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})
ACCENT   = "#a8f0c6"
ACCENT2  = "#7eb8f7"
ACCENT3  = "#f0a8d0"
PAL      = [ACCENT, ACCENT2, ACCENT3, "#f7c97e", "#b8a8f0", "#f07e7e"]

# ── Helpers 
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss  = df.isnull().sum()
    pct   = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"Missing Count": miss, "Missing %": pct, "Dtype": df.dtypes})[miss > 0].sort_values("Missing %", ascending=False)

# ── Sidebar 
with st.sidebar:
    st.markdown("## 🔬 DataLens")
    st.markdown("<span style='color:#7a7f94;font-size:0.78rem;'>CSV Explorer & Visualizer</span>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    st.markdown("---")
    st.markdown("<span style='color:#7a7f94;font-size:0.75rem;'>Supports UTF-8 encoded CSVs up to 200 MB</span>", unsafe_allow_html=True)

# ── Main 
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

df = load_data(uploaded)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# ── KPI row 
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",          f"{df.shape[0]:,}")
c2.metric("Columns",       f"{df.shape[1]:,}")
c3.metric("Numeric cols",  len(num_cols))
c4.metric("Categoric cols",len(cat_cols))
miss_total = df.isnull().sum().sum()
c5.metric("Missing cells", f"{miss_total:,}", delta=f"{miss_total/df.size*100:.1f}% of total", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋  Preview", "🩺  Missing Values", "🔗  Correlation", "📊  Visualisations"])

# ─── Tab 1 · Preview
with tab1:
    st.markdown("## Dataset Preview")
    n_rows = st.slider("Rows to display", 5, min(200, len(df)), 10, key="preview_rows")
    st.dataframe(df.head(n_rows), use_container_width=True, height=380)

    with st.expander("📐 Shape & Column Info"):
        buf = StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        st.code(info_str, language="")

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
            fig, ax = plt.subplots(figsize=(6, max(3, len(ms) * 0.45)))
            bars = ax.barh(ms.index, ms["Missing %"], color=ACCENT3, edgecolor="none", height=0.55)
            ax.set_xlabel("Missing  %")
            ax.invert_yaxis()
            ax.axvline(5, color="#f7c97e", lw=1, ls="--", alpha=0.7)
            ax.set_title("Missing values by column", pad=10)
            for bar, val in zip(bars, ms["Missing %"]):
                ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=8, color="#c9ccd6")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Heatmap of missingness
        st.markdown("### Missingness pattern heatmap")
        sample = df[ms.index].head(300).isnull().astype(int)
        fig2, ax2 = plt.subplots(figsize=(max(6, len(ms)*0.7), 3.5))
        sns.heatmap(sample.T, cmap=["#1f2330", ACCENT3], cbar=False,
                    linewidths=0, ax=ax2, yticklabels=True, xticklabels=False)
        ax2.set_xlabel("Row index (first 300)")
        ax2.set_title("White = missing  ·  Dark = present", pad=8)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

# ─── Tab 3 · Correlation 
with tab3:
    st.markdown("## Correlation Heatmap")
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns to compute correlations.")
    else:
        method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
        sel_cols = st.multiselect("Select columns", num_cols, default=num_cols[:min(12, len(num_cols))])

        if len(sel_cols) < 2:
            st.info("Select at least 2 columns.")
        else:
            corr = df[sel_cols].corr(method=method)
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

            fig, ax = plt.subplots(figsize=(max(6, len(sel_cols)*0.75), max(5, len(sel_cols)*0.65)))
            cmap = sns.diverging_palette(210, 150, s=80, l=40, as_cmap=True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                        linewidths=0.5, linecolor="#0d0f14",
                        annot_kws={"size": 8}, ax=ax,
                        vmin=-1, vmax=1,
                        cbar_kws={"shrink": 0.7, "pad": 0.02})
            ax.set_title(f"{method.capitalize()} correlation matrix", pad=12)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # Top pairs
            pairs = (corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                         .stack().reset_index()
                         .rename(columns={"level_0":"Col A","level_1":"Col B",0:"Correlation"})
                         .assign(AbsCorr=lambda x: x["Correlation"].abs())
                         .sort_values("AbsCorr", ascending=False)
                         .drop(columns="AbsCorr"))
            with st.expander("🔍 Top correlated pairs"):
                st.dataframe(pairs.head(15).style.background_gradient(subset=["Correlation"], cmap="RdYlGn"), use_container_width=True)

# ─── Tab 4 · Visualisations 
with tab4:
    st.markdown("## Visualisations")
    vtype = st.selectbox("Chart type", [
        "Histogram", "Box Plot", "Scatter Plot",
        "Bar Chart (categorical)", "Line Chart", "Pair Plot"
    ])

    # ── Histogram
    if vtype == "Histogram":
        col = st.selectbox("Numeric column", num_cols)
        bins = st.slider("Bins", 10, 100, 30)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[col].dropna(), bins=bins, color=ACCENT, edgecolor="#0d0f14", alpha=0.9)
        ax.set_xlabel(col); ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {col}")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    # ── Box Plot 
    elif vtype == "Box Plot":
        cols_sel = st.multiselect("Numeric columns", num_cols, default=num_cols[:min(6, len(num_cols))])
        if cols_sel:
            fig, ax = plt.subplots(figsize=(max(6, len(cols_sel)*1.2), 5))
            data_to_plot = [df[c].dropna().values for c in cols_sel]
            bp = ax.boxplot(data_to_plot, patch_artist=True, labels=cols_sel,
                            medianprops=dict(color="#0d0f14", linewidth=2))
            colors = PAL * (len(cols_sel)//len(PAL)+1)
            for patch, clr in zip(bp["boxes"], colors):
                patch.set_facecolor(clr); patch.set_alpha(0.85)
            ax.set_title("Box plots"); plt.xticks(rotation=30, ha="right")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    # ── Scatter Plot 
    elif vtype == "Scatter Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            cx, cy = st.columns(2)
            x_col  = cx.selectbox("X axis", num_cols, index=0)
            y_col  = cy.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
            hue_col = st.selectbox("Colour by (optional)", ["None"] + cat_cols + num_cols)
            sample_n = st.slider("Max points", 100, min(5000, len(df)), min(1000, len(df)))
            plot_df = df[[x_col, y_col] + ([hue_col] if hue_col != "None" else [])].dropna()
            subset = plot_df.sample(min(sample_n, len(plot_df)), random_state=42)
            
            if hue_col == "None":
                ax.scatter(subset[x_col], subset[y_col], color=ACCENT, alpha=0.5, s=20, edgecolors="none")
            else:
                cats = subset[hue_col].astype(str).unique()
                for i, cat in enumerate(cats):
                    mask = subset[hue_col].astype(str) == cat
                    ax.scatter(subset.loc[mask, x_col], subset.loc[mask, y_col],
                               color=PAL[i % len(PAL)], label=cat, alpha=0.55, s=20, edgecolors="none")
                ax.legend(fontsize=7, framealpha=0.2, loc="best")
            ax.set_xlabel(x_col); ax.set_ylabel(y_col)
            ax.set_title(f"{x_col}  vs  {y_col}")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    # ── Bar Chart 
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

    # ── Line Chart 
    elif vtype == "Line Chart":
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            y_cols  = st.multiselect("Y columns", num_cols, default=num_cols[:min(3, len(num_cols))])
            x_opts  = ["Row index"] + df.columns.tolist()
            x_col   = st.selectbox("X axis", x_opts)
            if y_cols:
                fig, ax = plt.subplots(figsize=(10, 4))
                for i, col in enumerate(y_cols):
                    x_vals = df.index if x_col == "Row index" else df[x_col]
                    ax.plot(x_vals, df[col], color=PAL[i % len(PAL)], lw=1.5, label=col, alpha=0.9)
                ax.legend(fontsize=8, framealpha=0.2)
                ax.set_xlabel(x_col if x_col != "Row index" else "Index")
                ax.set_title("Line chart")
                fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    # ── Pair Plot 
    elif vtype == "Pair Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            max_cols = st.multiselect("Columns (max 5 recommended)", num_cols, default=num_cols[:min(4, len(num_cols))])
            if len(max_cols) >= 2:
                sample_pp = df[max_cols].dropna().sample(min(500, len(df)), random_state=0)
                fig = sns.pairplot(sample_pp, plot_kws={"alpha":0.45, "s":12, "color": ACCENT},
                                   diag_kws={"color": ACCENT2, "alpha":0.8})
                fig.figure.patch.set_facecolor("#13161e")
                for ax_row in fig.axes:
                    for ax_ in ax_row:
                        ax_.set_facecolor("#13161e")
                        for spine in ax_.spines.values():
                            spine.set_edgecolor("#1f2330")
                st.pyplot(fig.figure, use_container_width=True)
