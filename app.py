from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cross-Asset Market Topology & Systemic Stress Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌐 Cross-Asset Market Topology & Systemic Stress Explorer")
st.markdown(
    "Quantitative framework evaluating market fragility, topological regime shifts, "
    "and Random Matrix Theory (RMT) eigenvalue noise filtering across macro assets."
)

# ---------------------------------------------------------
# Data Loader
# ---------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "macro_log_returns.parquet"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Data file not found at `{DATA_PATH}`. Please run `python src/fetch_macro_data.py` first.")
        st.stop()
    return pd.read_parquet(DATA_PATH)

returns_df = load_data()

DISPLAY_LABELS = {
    "SP500_Equity": "S&P 500\n(Equities)",
    "Crude_Oil": "Crude Oil\n(Macro Hub)",
    "Gold": "Gold\n(Defensive)",
    "US_Dollar_Index": "Dollar Index\n(Currency)",
    "US_10Y_Treasury": "10Y Yield\n(Duration)",
    "Bitcoin": "Bitcoin\n(Liquidity Beta)",
}

ASSET_COLORS = {
    "Bitcoin": "#f7931a",          # Bitcoin Orange
    "SP500_Equity": "#059669",      # Equity Emerald
    "Crude_Oil": "#475569",         # Energy Slate
    "Gold": "#d97706",              # Precious Amber
    "US_Dollar_Index": "#2563eb",   # Currency Blue
    "US_10Y_Treasury": "#7c3aed",   # Rates Violet
}

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Network Parameters")

all_assets = returns_df.columns.tolist()
selected_assets = st.sidebar.multiselect(
    "Select Assets for Analysis",
    options=all_assets,
    default=all_assets,
    format_func=lambda x: DISPLAY_LABELS.get(x, x).replace("\n", " "),
)

if len(selected_assets) < 3:
    st.warning("Please select at least 3 assets to construct a valid Minimum Spanning Tree.")
    st.stop()

use_rmt = st.sidebar.toggle("Apply RMT Noise Filtering (Marčenko-Pastur)", value=False)
window = st.sidebar.slider("Rolling Window Size (Trading Days)", min_value=30, max_value=120, value=60, step=5)

filtered_returns = returns_df[selected_assets]

# ---------------------------------------------------------
# Quantitative Machinery
# ---------------------------------------------------------
def clean_correlation_matrix(corr: pd.DataFrame, T: int) -> pd.DataFrame:
    """Marčenko-Pastur spectral cleaning using trace-preserving clumping."""
    N = corr.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(corr.values)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    q = N / T
    lambda_plus = (1.0 + np.sqrt(q)) ** 2
    
    cleaned_evals = eigenvalues.copy()
    noise_mask = cleaned_evals <= lambda_plus
    
    if np.any(noise_mask):
        cleaned_evals[noise_mask] = np.mean(cleaned_evals[noise_mask])
        
    corr_clean = eigenvectors @ np.diag(cleaned_evals) @ eigenvectors.T
    inv_diag = np.diag(1.0 / np.sqrt(np.diag(corr_clean)))
    corr_clean = inv_diag @ corr_clean @ inv_diag
    return pd.DataFrame(corr_clean, index=corr.index, columns=corr.columns).clip(-1.0, 1.0)

def compute_mst(corr_matrix: pd.DataFrame):
    """Calculates Mantegna ultrametric distance and extracts Kruskal MST."""
    dist_matrix = np.sqrt(2.0 * (1.0 - corr_matrix.clip(-1.0, 1.0)))
    G = nx.Graph()
    cols = dist_matrix.columns.tolist()
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            G.add_edge(cols[i], cols[j], weight=dist_matrix.iloc[i, j])
            
    return nx.minimum_spanning_tree(G, algorithm="kruskal"), dist_matrix

# Compute Static Baseline Network
T_full = len(filtered_returns)
corr_full = filtered_returns.corr()

if use_rmt:
    corr_processed = clean_correlation_matrix(corr_full, T_full)
    tree_title = "RMT-Filtered MST (Marčenko-Pastur Denoised)"
    tree_subtitle = "Denoised spectral modes isolate structural cross-asset backbone"
else:
    corr_processed = corr_full
    tree_title = "Global Cross-Asset Network Topology (2021–2025)"
    tree_subtitle = r"Mantegna Minimum Spanning Tree ($d_{ij} = \sqrt{2(1 - \rho_{ij})}$) | Distances closer to 0 represent tighter co-movement"

mst, dist_df = compute_mst(corr_processed)

# ---------------------------------------------------------
# Figure 1: Exact Notebook Matplotlib MST
# ---------------------------------------------------------
st.subheader("1. Market Backbone & Metric Distances")

node_colors = [ASSET_COLORS.get(node, "#3b82f6") for node in mst.nodes()]
node_labels = {node: DISPLAY_LABELS.get(node, node) for node in mst.nodes()}

fig1, ax1 = plt.subplots(figsize=(13, 8), facecolor="#f8fafc")
ax1.set_facecolor("#f8fafc")

pos = nx.spring_layout(mst, k=2.2, seed=42)

edges = list(mst.edges(data=True))
edge_weights = [d["weight"] for _, _, d in edges]
edge_widths = [4.5 * (1.8 - w) for w in edge_weights]

# Edges
nx.draw_networkx_edges(
    mst,
    pos,
    ax=ax1,
    width=edge_widths,
    edge_color="#64748b",
    alpha=0.6,
    style="solid",
)

# Nodes
nx.draw_networkx_nodes(
    mst,
    pos,
    ax=ax1,
    node_size=4500,
    node_color=node_colors,
    edgecolors="#ffffff",
    linewidths=3,
    alpha=0.95,
)

# Labels inside nodes
nx.draw_networkx_labels(
    mst,
    pos,
    labels=node_labels,
    ax=ax1,
    font_size=8,
    font_family="sans-serif",
    font_weight="bold",
    font_color="#ffffff",
)

# Badges with distance and implied correlation
edge_labels = {
    (u, v): f"d = {d['weight']:.2f}\n(ρ ≈ {1.0 - (d['weight']**2)/2.0:+.2f})"
    for u, v, d in edges
}

nx.draw_networkx_edge_labels(
    mst,
    pos,
    edge_labels=edge_labels,
    ax=ax1,
    font_size=8,
    font_color="#0f172a",
    bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#cbd5e1", lw=1.2, alpha=0.9),
)

ax1.set_title(tree_title, fontsize=16, fontweight="bold", pad=25, color="#0f172a")
ax1.text(0.5, 1.02, tree_subtitle, transform=ax1.transAxes, ha="center", fontsize=10, color="#475569")
ax1.axis("off")
plt.tight_layout()

st.pyplot(fig1)
plt.close(fig1)

# ---------------------------------------------------------
# Figure 2: Exact Notebook Two-Panel Stress Chart
# ---------------------------------------------------------
st.subheader("2. Dynamic Systemic Stress Index & Macro Regimes")

@st.cache_data
def calculate_rolling_dynamics(returns_subset: pd.DataFrame, win: int, rmt_flag: bool):
    dates_list = []
    ttl_list = []
    btc_spy_dist = []
    tickers = returns_subset.columns.tolist()
    
    for i in range(win, len(returns_subset)):
        sub_df = returns_subset.iloc[i - win : i]
        current_date = returns_subset.index[i]
        corr_local = sub_df.corr().clip(-1.0, 1.0)
        
        if rmt_flag:
            corr_local = clean_correlation_matrix(corr_local, win)
            
        tree_local, dist_local = compute_mst(corr_local)
        total_len = sum(d["weight"] for _, _, d in tree_local.edges(data=True))
        
        dates_list.append(current_date)
        ttl_list.append(total_len)
        
        if "Bitcoin" in tickers and "SP500_Equity" in tickers:
            btc_spy_dist.append(dist_local.loc["Bitcoin", "SP500_Equity"])
        else:
            btc_spy_dist.append(np.nan)
            
    res_df = pd.DataFrame(
        {"total_tree_length": ttl_list, "btc_spy_distance": btc_spy_dist},
        index=pd.to_datetime(dates_list),
    )
    return res_df

with st.spinner("Calculating rolling network dynamics across historical regimes..."):
    stress_df = calculate_rolling_dynamics(filtered_returns, window, use_rmt)

fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, facecolor="#f8fafc")
ax2_1.set_facecolor("#f8fafc")
ax2_2.set_facecolor("#f8fafc")

# Panel 1: Total Tree Length L(t)
ax2_1.plot(stress_df.index, stress_df["total_tree_length"], color="#dc2626", lw=1.8, label="Total Tree Length L(t)")
mean_ttl = stress_df["total_tree_length"].mean()
ax2_1.axhline(mean_ttl, color="#475569", linestyle="--", alpha=0.7, label=f"Historical Mean ({mean_ttl:.2f})")
ax2_1.set_title(f"Cross-Asset Systemic Stress Index (Rolling {window}-Day MST)", fontsize=14, fontweight="bold", color="#0f172a")
ax2_1.set_ylabel("Tree Length\n(Contraction = Stress)", fontsize=11, color="#0f172a")
ax2_1.grid(True, linestyle=":", alpha=0.6)
ax2_1.legend(loc="upper left")

# Panel 2: Bitcoin vs S&P 500 Dynamic Coupling
has_btc_spy = not stress_df["btc_spy_distance"].isna().all()
if has_btc_spy:
    ax2_2.plot(stress_df.index, stress_df["btc_spy_distance"], color="#f7931a", lw=1.8, label="Bitcoin vs S&P 500 Distance")
    ax2_2.axhline(np.sqrt(2.0), color="#2563eb", linestyle="--", alpha=0.7, label=r"Orthogonal Threshold ($d = \sqrt{2}, \rho = 0$)")
    ax2_2.set_title("Bitcoin Coupling to Traditional Equities", fontsize=12, fontweight="bold", color="#0f172a")
    ax2_2.set_ylabel("Mantegna Distance", fontsize=11, color="#0f172a")
    ax2_2.set_xlabel("Date", fontsize=11, color="#0f172a")
    ax2_2.grid(True, linestyle=":", alpha=0.6)
    ax2_2.legend(loc="upper left")
else:
    ax2_2.text(0.5, 0.5, "Select both 'Bitcoin' and 'S&P 500' to view coupling dynamics.", ha="center", va="center", transform=ax2_2.transAxes)

plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ---------------------------------------------------------
# Scorecard Metrics
# ---------------------------------------------------------
curr_val = stress_df["total_tree_length"].iloc[-1]
delta_mean = curr_val - mean_ttl

col1, col2, col3 = st.columns(3)
col1.metric("Current Total Tree Length", f"{curr_val:.3f}")
col2.metric("Historical Mean Baseline", f"{mean_ttl:.3f}")
col3.metric(
    "Systemic Regime State",
    "Decoupled / Normal" if delta_mean >= 0 else "Contraction / Elevated Stress",
    delta=f"{delta_mean:.3f}",
    delta_color="normal" if delta_mean >= 0 else "inverse",
)