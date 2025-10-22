import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from networkx.algorithms.community import greedy_modularity_communities


def feature_selection_graph(df_features: pd.DataFrame, correlation_threshold: float = 0.7):
    """
    Select features via a correlation graph + PageRank per community.

    Steps
    -----
    1) Standardize numeric features.
    2) Build an undirected graph where edges connect feature pairs with
       |corr| > correlation_threshold (edge weight = |corr|).
    3) Detect communities by greedy modularity.
    4) Pick the highest-PageRank feature within each community.

    Parameters
    ----------
    df_features : pd.DataFrame
        Each column is a feature (numeric). Non-numeric columns are ignored.
    correlation_threshold : float, default=0.7
        Absolute Pearson correlation above which an edge is added.

    Returns
    -------
    selected_features : list[str]
        One representative feature per community (max PageRank).
    G : networkx.Graph
        The correlation graph with edge attribute 'weight' = |corr|.

    Notes
    -----
    - Constant-variance columns are dropped before computing correlations.
    - If no edges satisfy the threshold, each feature forms its own community.
    - PageRank on an undirected weighted graph is well-defined in NetworkX.
    """
    # --- Validate / prepare data
    if not (0 <= correlation_threshold < 1):
        raise ValueError("correlation_threshold must be in [0, 1).")

    X = df_features.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found in df_features.")

    # Drop constant columns (avoid NaN correlations)
    nunique = X.nunique(dropna=True)
    constant_cols = nunique.index[nunique <= 1]
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        # All columns were constant; return empty selection and empty graph
        return [], nx.Graph()

    # --- Standardize (not strictly required for Pearson, but fine)
    scaler = StandardScaler()
    Xz = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # --- Correlation matrix
    corr = Xz.corr().fillna(0.0)  # safety

    # --- Build graph (vectorized upper-triangular thresholding)
    feats = X.columns.to_list()
    G = nx.Graph()
    G.add_nodes_from(feats)

    cvals = corr.values
    p = cvals.shape[0]
    mask = np.triu(np.ones((p, p), dtype=bool), k=1)
    strong = (np.abs(cvals) > correlation_threshold) & mask
    rows, cols = np.where(strong)

    for i, j in zip(rows, cols):
        G.add_edge(feats[i], feats[j], weight=float(abs(cvals[i, j])))

    # --- Communities
    if G.number_of_edges() > 0:
        communities = greedy_modularity_communities(G, weight='weight')
    else:
        # No edges: each node is its own community
        communities = [frozenset([f]) for f in feats]

    # --- PageRank (weights = abs corr)
    pagerank_scores = nx.pagerank(G, weight='weight') if G.number_of_nodes() > 0 else {}

    # --- Select one feature per community (max PageRank)
    selected_features = []
    for comm in communities:
        best = max(comm, key=lambda x: pagerank_scores.get(x, 0.0))
        selected_features.append(best)

    return selected_features, G