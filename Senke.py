# sankey_app.py
# Streamlit app to visualize a fixed Excel lineage file as:
# 1) A Sankey diagram
# 2) A neural-net-style force-directed graph
# 3) A pipeline-style DAG view
#
# The Excel file must be located at: data/Choice Data to Visualize.xlsx

from pathlib import Path
import io
import re

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- App Config ----------
st.set_page_config(page_title="Choice Data Sankey and Graph Views", layout="wide")
st.title("Choice Data Dependency Visualizer")

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

    # Aggregate duplicate Source -> Target links by frequency
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

    # Build undirected graph to emphasize connectedness
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


def build_dag(df: pd.DataFrame, aggregate: bool, min_count: int):
    """
    Build a pipeline-style DAG view:
    - Nodes are arranged in vertical columns by Stage (Upstream, INT, ACC, etc).
    - Edges are directed from left to right.
    Returns: (figure, node_count, edge_count)
    """
    data = df.copy()
    if aggregate:
        data["Source"] = data["Source"].apply(group_source)

    # Keep Stage for the target so we can column-position nodes
    link_counts = (
        data.groupby(["Source", "Target", "Stage"], as_index=False)
        .size()
        .rename(columns={"size": "Value"})
    )
    link_counts = link_counts[link_counts["Value"] >= min_count]
    if link_counts.empty:
        return go.Figure(), 0, 0

    # Determine a stage for each node
    target_stage_map = (
        link_counts[link_counts["Stage"] != ""]
        .groupby("Target")["Stage"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    nodes = set(link_counts["Source"]).union(set(link_counts["Target"]))
    node_stage = {}
    for n in nodes:
        if n in target_stage_map:
            node_stage[n] = target_stage_map[n]
        else:
            node_stage[n] = "Upstream"

    # Order the stages into columns
    known_order = ["Upstream", "RAW", "STG", "INT", "ACC", "CORE"]
    stages_present = list({node_stage[n] for n in nodes})
    ordered = []
    for s in known_order:
        if s in stages_present:
            ordered.append(s)
    for s in stages_present:
        if s not in ordered:
            ordered.append(s)
    stage_to_x = {stage: i for i, stage in enumerate(ordered)}
    max_x = max(stage_to_x.values()) if stage_to_x else 0

    # Compute node positions: x is column, y is spaced within column
    nodes_by_stage = {}
    for n, stg in node_stage.items():
        nodes_by_stage.setdefault(stg, []).append(n)

    node_pos = {}
    for stage, stage_nodes in nodes_by_stage.items():
        x = 0 if max_x == 0 else stage_to_x[stage] / max_x
        count = len(stage_nodes)
        if count == 1:
            ys = [0.5]
        else:
            ys = [i / (count - 1) for i in range(count)]
        for n, y in zip(stage_nodes, ys):
            node_pos[n] = (x, y)

    # Edge traces
    edge_x = []
    edge_y = []
    for _, row in link_counts.iterrows():
        src = row["Source"]
        tgt = row["Target"]
        x0, y0 = node_pos[src]
        x1, y1 = node_pos[tgt]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="rgba(150,150,150,0.7)"),
        hoverinfo="none",
        mode="lines",
    )

    # Directed graph to compute weighted degree
    G = nx.DiGraph()
    for _, row in link_counts.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=row["Value"])

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for n in nodes:
        x, y = node_pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{n} ({node_stage[n]})")
        deg = G.degree(n, weight="weight")
        node_size.append(12 + deg * 1.5)

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
        title="Pipeline dependency DAG",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig, len(nodes), len(link_counts)


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
    view_type = st.radio("View type", ["Sankey", "Neural graph", "DAG (pipeline)"], index=0)
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

# ---------- Build and Render ----------
if view_type == "Sankey":
    fig, node_count, link_count = build_sankey(
        filtered,
        aggregate=(aggregate == "Group sources by system"),
        min_count=min_count,
    )
elif view_type == "Neural graph":
    fig, node_count, link_count = build_graph(
        filtered,
        aggregate=(aggregate == "Group sources by system"),
        min_count=min_count,
    )
else:  # DAG (pipeline)
    fig, node_count, link_count = build_dag(
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
