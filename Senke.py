# sankey_app.py
# Streamlit app to visualize a fixed Excel lineage file as:
# 1) A Sankey diagram
# 2) A neural-net-style force-directed graph
#
# The Excel file must be located at: data/Choice Data to Visualize.xlsx

from pathlib import Path
import io
import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

# ---------- App Config ----------
st.set_page_config(page_title="Choice Data Sankey", layout="wide")
st.title("Choice Data Sankey And Graph View")

# Location of the fixed Excel file, bundled with the app repo
DATA_PATH = Path(__file__).parent / "data" / "Choice Data to Visualize.xlsx"

REQUIRED_COLS = [
    "Source Table/Dependency",
    "Target Table Name",
    "Stage",
    "Scheduling Group/Airflow Batch",
]

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_bytes_from_repo(path: Path) -> bytes:
    """Read the Excel file bytes from the repo path."""
    return path.read_bytes()

@st.cache_data(show_spinner=False)
def load_and_expand_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Load Excel and expand rows so each comma-separated source becomes its own link row.
    Ensures required columns exist. Returns a tidy DataFrame with Source, Target, Stage, Batch.
    """
    df = pd.read_excel(io.BytesIO(file_bytes)).fillna("")
    # Ensure expected columns exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""

    rows = []
    for _, r in df.iterrows():
        raw_sources = str(r["Source Table/Dependency"])
        target = str(r["Target Table Name"]).strip()
        sources = [s.strip() for s in raw_sources.split(",") if s.strip()] or ["(unknown)"]
        for s in sources:
            rows.append(
                {
                    "Source": s,
                    "Target": target,
                    "Stage": str(r["Stage"]).strip(),
                    "Batch": str(r["Scheduling Group/Airflow Batch"]).strip(),
                }
            )
    return pd.DataFrame(rows)

def group_source(name: str) -> str:
    """
    Collapse noisy node names into a higher-level bucket (useful when aggregating).
    Examples:
      JDE_F0005 -> JDE
      STG_F0005_CODE_AREA_CLASSIFICATION -> STG
      RETOOL_ACTION_ASSIGNMENT -> RETOOL
    Fallback: token before first underscore if meaningful.
    """
    m = re.match(r"^(JDE|STG|RETOOL|RAW|INT|CORE|CODE|ETL|DWH|DIM|FACT)\b.*", name, flags=re.I)
    if m:
        return m.group(1).upper()
    t = name.split("_")[0].upper()
    return t if len(t) >= 3 else name

def build_sankey(df: pd.DataFrame, aggregate: bool, min_count: int):
    """
    Build a Sankey figure from a tidy DataFrame with columns: Source, Target.
    - aggregate: if True, group sources by system (for example JDE, STG, RETOOL)
    - min_count: drop links with frequency below this threshold
    Returns: (figure, node_count, link_count)
    """
    data = df.copy()
    if aggregate:
        data["Source"] = data["Source"].apply(group_source)

    # Aggregate duplicate Source->Target links by frequency
    link_counts = (
        data.groupby(["Source", "Target"], as_index=False)
            .size()
            .rename(columns={"size": "Value"})
    )

    # Filter small links
    link_counts = link_counts[link_counts["Value"] >= min_count]
    if link_counts.empty:
        return go.Figure(), 0, 0

    sources = link_counts["Source"]
    targets = link_counts["Target"]
    nodes = pd.Index(pd.unique(pd.concat([sources, targets], ignore_index=True))).tolist()
    idx = {n: i for i, n in enumerate(nodes)}

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=nodes,
                    pad=16,
                    thickness=16,
                    line=dict(color="black", width=0.5),
                ),
                link=dict(
                    source=[idx[s] for s in sources],
                    target=[idx[t] for t in targets],
                    value=link_counts["Value"].tolist(),
                    hovertemplate=" %{source.label} → %{target.label}<br>Count: %{value}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig, len(nodes), len(link_counts)

def build_graph(df: pd.DataFrame, aggregate: bool, min_count: int):
    """
    Build a neural-net-style force-directed graph using NetworkX and Plotly.
    Nodes are arranged with a spring layout and sized by weighted degree.
    Returns: (figure, node_count, edge_count)
    """
    data = df.copy()
    if aggregate:
        data["Source"] = data["Source"].apply(group_source)

    link_counts = (
        data.groupby(["Source", "Target"], as_index=False)
            .size()
            .rename(columns={"size": "Value"})
    )
    link_counts = link_counts[link_counts["Value"] >= min_count]
    if link_counts.empty:
        return go.Figure(), 0, 0

    # Build graph
    G = nx.Graph()
    for _, row in link_counts.iterrows():
        src = row["Source"]
        tgt = row["Target"]
        w = row["Value"]
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += w
        else:
            G.add_edge(src, tgt, weight=w)

    # Spring layout positions
    pos = nx.spring_layout(G, k=0.7, iterations=60, seed=42)

    # Edges
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="rgba(150,150,150,0.7)"),
        hoverinfo="none",
        mode="lines",
    )

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        degree = G.degree(node, weight="weight")
        node_size.append(10 + degree * 2)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            size=node_size,
            line=dict(width=1, color="black"),
            color="rgba(31,119,180,0.9)",
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig, G.number_of_nodes(), G.number_of_edges()

# ---------- Load fixed file ----------
if not DATA_PATH.exists():
    st.error("The fixed Excel file is missing. Expected at: data/Choice Data to Visualize.xlsx")
    st.stop()

file_bytes = load_bytes_from_repo(DATA_PATH)
df = load_and_expand_from_bytes(file_bytes)

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.header("Filters")
    stages = sorted([s for s in df["Stage"].unique() if s])
    batches = sorted([b for b in df["Batch"].unique() if b])

    stage_sel = st.multiselect("Stage", stages, default=stages or [])
    batch_sel = st.multiselect("Scheduling Group / Airflow Batch", batches, default=batches or [])
    text_query = st.text_input("Search (matches Source or Target)", "")
    view_type = st.radio("View type", ["Sankey", "Neural graph"], index=0)
    aggregate = st.radio("Aggregation level", ["Raw nodes", "Group sources by system"], index=1)
    min_count = st.slider("Hide links with count less than", 1, 10, 1, 1)

# Apply filters
filtered = df.copy()
if stage_sel:
    filtered = filtered[filtered["Stage"].isin(stage_sel) | (filtered["Stage"] == "")]
if batch_sel:
    filtered = filtered[filtered["Batch"].isin(batch_sel) | (filtered["Batch"] == "")]
if text_query.strip():
    q = text_query.strip().lower()
    mask = filtered["Source"].str.lower().str.contains(q) | filtered["Target"].str.lower().str.contains(q)
    filtered = filtered[mask]

# ---------- Build & Render ----------
if view_type == "Sankey":
    fig, node_count, link_count = build_sankey(
        filtered,
        aggregate=(aggregate == "Group sources by system"),
        min_count=min_count,
    )
else:
    fig, node_count, link_count = build_graph(
        filtered,
        aggregate=(aggregate == "Group sources by system"),
        min_count=min_count,
    )

st.caption(f"Nodes: {node_count} | Links: {link_count}")
st.plotly_chart(fig, use_container_width=True)

# ---------- Download Edges ----------
edges_csv = (
    filtered.groupby(["Source", "Target"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
            .sort_values("Count", ascending=False)
)
st.download_button(
    "Download current edges as CSV",
    data=edges_csv.to_csv(index=False).encode("utf-8"),
    file_name="sankey_filtered_edges.csv",
    mime="text/csv",
)

st.caption("Data source: data/Choice Data to Visualize.xlsx")
