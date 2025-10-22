import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from typing import Iterable, List, Tuple


def degree_distribution(G) -> Tuple[np.ndarray, np.ndarray]:
    """
    Degree distribution P(k) of a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph. For directed graphs, NetworkX degree() = in+out.

    Returns
    -------
    kvalues : np.ndarray
        Degrees 0..k_max appearing in the graph.
    Pk : np.ndarray
        Probability mass function over kvalues (sum == 1).
    """
    deg = np.fromiter((d for _, d in G.degree()), dtype=int)
    if deg.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    counts = np.bincount(deg)
    Pk = counts / counts.sum()
    kvalues = np.arange(counts.size)
    return kvalues, Pk


def shannon_entropy(G, base: float = 2.0) -> float:
    """
    Shannon entropy of the degree distribution.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
    base : float, default=2.0
        Log base (2=bits, np.e or None=nats).

    Returns
    -------
    float
        H = -∑ P(k) log_base P(k). 0.0 for empty graphs.
    """
    _, Pk = degree_distribution(G)
    if Pk.size == 0:
        return 0.0
    p = Pk[Pk > 0]
    if base == 2:
        return float(-(p * np.log2(p)).sum())
    elif base in (None, np.e):
        return float(-(p * np.log(p)).sum())
    else:
        return float(-(p * np.log(p) / np.log(base)).sum())


def compute_mutual_information(
    degree_seq1: Iterable, degree_seq2: Iterable, base: float = 2.0
) -> float:
    """
    Mutual information I(X;Y) between two discrete sequences (exact contingency).

    Parameters
    ----------
    degree_seq1, degree_seq2 : Iterable
        Equal-length sequences (e.g., node degrees per layer).
    base : float, default=2.0
        Log base.

    Returns
    -------
    float
        Mutual information in chosen units.

    Notes
    -----
    Uses np.unique(..., return_inverse=True) to remap values to 0..K-1 and builds
    the joint table via np.add.at for exact co-occurrences (no binning).
    """
    x = np.asarray(degree_seq1)
    y = np.asarray(degree_seq2)
    if x.size != y.size:
        raise ValueError("Sequences must have same length.")

    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    Kx, Ky = xi.max() + 1, yi.max() + 1

    joint = np.zeros((Kx, Ky), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)

    N = joint.sum()
    if N == 0:
        return 0.0

    pxy = joint / N
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = pxy > 0

    mi = np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask]))
    if base == 2:
        mi /= np.log(2)
    elif base not in (None, np.e):
        mi /= np.log(base)
    return float(mi)


def compute_layer_correlation_matrix(adj_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Inter-layer *mutual information* matrix based on node degrees.

    Parameters
    ----------
    adj_matrices : list of np.ndarray
        M adjacency matrices, each N x N. Degree is computed as row-sum (out-degree
        for directed graphs; strength for weighted).

    Returns
    -------
    np.ndarray
        M x M symmetric matrix of MI (bits). Diagonal zeros.

    Notes
    -----
    If you need in-degree instead, use A.sum(axis=0). For binary degrees on
    weighted graphs, use (A > 0).sum(axis=1).
    """
    M = len(adj_matrices)
    if M == 0:
        return np.zeros((0, 0), dtype=float)
    degs = [np.sum(A, axis=1) for A in adj_matrices]
    mi = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            val = compute_mutual_information(degs[i], degs[j], base=2.0)
            mi[i, j] = mi[j, i] = val
    return mi


def average_edge_overlap(
    adj_matrices: List[np.ndarray], undirected: bool = True, binary: bool = True
) -> np.ndarray:
    """
    Average edge overlap (Jaccard of edge sets) ω_{αβ} = |E_α ∩ E_β| / |E_α ∪ E_β|.

    Parameters
    ----------
    adj_matrices : list of np.ndarray
        M adjacency matrices, each N x N.
    undirected : bool, default=True
        If True, count only the upper triangle with k=1 (exclude diagonal).
    binary : bool, default=True
        If True, threshold edges at >0 before set operations.

    Returns
    -------
    np.ndarray
        M x M symmetric matrix with values in [0, 1]. Diagonal zeros.
    """
    M = len(adj_matrices)
    if M == 0:
        return np.zeros((0, 0), dtype=float)

    omega = np.zeros((M, M), dtype=float)
    for a in range(M):
        A = adj_matrices[a]
        A = (A > 0).astype(int) if binary else A
        A_use = np.triu(A, k=1) if undirected else A
        A_bin = (A_use > 0)

        for b in range(a + 1, M):
            B = adj_matrices[b]
            B = (B > 0).astype(int) if binary else B
            B_use = np.triu(B, k=1) if undirected else B
            B_bin = (B_use > 0)

            inter = np.sum(A_bin & B_bin)
            uni = np.sum(A_bin | B_bin)
            omega_val = inter / uni if uni > 0 else 0.0
            omega[a, b] = omega[b, a] = omega_val
    return omega


def graph_mutual_information(A: np.ndarray, B: np.ndarray, base: float = 2.0) -> float:
    """
    MI between two graphs A and B via their node degree sequences.

    Parameters
    ----------
    A, B : np.ndarray
        N x N adjacency matrices (binary or weighted).
    base : float, default=2.0
        Log base.

    Returns
    -------
    float
        Mutual information I(deg(A); deg(B)).
    """
    kA = A.sum(axis=1)
    kB = B.sum(axis=1)
    return compute_mutual_information(kA, kB, base=base)


def multiplex_graph_mutual_information(multiplex_graph: List[np.ndarray]) -> np.ndarray:
    """
    Pairwise MI between all layers of a multiplex graph (degree-based).

    Parameters
    ----------
    multiplex_graph : list of np.ndarray
        List of M adjacency matrices, each N x N.

    Returns
    -------
    np.ndarray
        M x M symmetric matrix of MI (bits).
    """
    M = len(multiplex_graph)
    if M == 0:
        return np.zeros((0, 0), dtype=float)
    mi = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            val = graph_mutual_information(multiplex_graph[i], multiplex_graph[j], base=2.0)
            mi[i, j] = mi[j, i] = val
    return mi


def toKNNAdjMatrix(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Convert an N×N *similarity* DataFrame into a kNN adjacency (0/1) DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        N×N similarity matrix (index == columns).
    k : int
        Number of neighbors for kNN.

    Returns
    -------
    adjacency_df : pd.DataFrame
        N×N binary adjacency. adjacency_df[i, j] = 1 if j is among the kNN of i
        OR i is among the kNN of j (symmetrized).
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("df must be square (N×N).")
    if not df.index.equals(df.columns):
        raise ValueError("df.index and df.columns must match and be aligned.")

    # Convert similarity to distance in [0, ∞). Ensure no NaNs.
    dist_df = (1.0 - df).fillna(1.0)
    np.fill_diagonal(dist_df.values, 0.0)

    # kNN graph on precomputed distances (no self-edges)
    knn_sparse = kneighbors_graph(
        dist_df,
        n_neighbors=k,
        metric='precomputed',
        mode='connectivity',
        include_self=False
    )

    # Symmetrize (undirected)
    knn_sparse_sym = knn_sparse.maximum(knn_sparse.T)

    adjacency_df = pd.DataFrame(
        knn_sparse_sym.toarray().astype(int),
        index=df.index,
        columns=df.columns
    )
    return adjacency_df


def toEpsilonNN(df: pd.DataFrame, epsilon: float, is_distance: bool = False, include_self: bool = False) -> pd.DataFrame:
    """
    Build an ε-neighborhood adjacency from an N×N distance or similarity matrix.

    Parameters
    ----------
    df : pd.DataFrame
        N×N matrix (index == columns). If is_distance=False, interpreted as similarity.
    epsilon : float
        Distance threshold: connect (i, j) if distance(i, j) ≤ epsilon.
    is_distance : bool, default=False
        If False, convert similarity → distance via (1 - df). If True, use df as distances.
    include_self : bool, default=False
        If True, keep self-edges on the diagonal.

    Returns
    -------
    adjacency_df : pd.DataFrame
        N×N binary, symmetrized adjacency.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("df must be square (N×N).")
    if not df.index.equals(df.columns):
        raise ValueError("df.index and df.columns must match and be aligned.")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    if is_distance:
        dist_df = df.copy().astype(float)
    else:
        dist_df = (1.0 - df).astype(float)

    dist_df = dist_df.fillna(np.inf)  # missing distances → not neighbors
    np.fill_diagonal(dist_df.values, 0.0 if include_self else np.inf)

    adjacency_sparse = radius_neighbors_graph(
        dist_df,
        radius=epsilon,
        metric='precomputed',
        mode='connectivity',
        include_self=include_self
    )

    # Symmetrize
    adjacency_sparse_sym = adjacency_sparse.maximum(adjacency_sparse.T)

    adjacency_df = pd.DataFrame(
        adjacency_sparse_sym.toarray().astype(int),
        index=df.index,
        columns=df.columns
    )
    return adjacency_df

def cosine_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between rows.

    Parameters
    ----------
    df : pd.DataFrame
        Each row is a feature vector.

    Returns
    -------
    sim_df : pd.DataFrame
        N×N similarity matrix with sim_df[i, j] = cosine(row_i, row_j).
    """
    sim = cosine_similarity(df)
    return pd.DataFrame(sim, index=df.index, columns=df.index)

def plot_two_years_overlap(df: pd.DataFrame, item: str, year1: int, year2: int):
    """
    Plot `item` for year1 and year2 on the same Day-of-Year axis.

    Parameters
    ----------
    df : pd.DataFrame
        DateTimeIndex; columns = items.
    item : str
        Column to plot.
    year1, year2 : int
        Calendar years to compare.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DateTimeIndex.")
    if item not in df.columns:
        raise KeyError(f"Column '{item}' not found in df.")

    df_y1 = df.loc[str(year1), [item]].copy()
    df_y2 = df.loc[str(year2), [item]].copy()

    df_y1['day_of_year'] = df_y1.index.dayofyear
    df_y2['day_of_year'] = df_y2.index.dayofyear

    plt.figure(figsize=(10, 5))
    plt.plot(df_y1['day_of_year'], df_y1[item], label=str(year1), alpha=0.8)
    plt.plot(df_y2['day_of_year'], df_y2[item], label=str(year2), alpha=0.8)
    plt.legend()
    plt.title(f'{item} — {year1} vs {year2}')
    plt.xlabel('Day of Year')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def gerar_matriz_adjacencia(cluster_dict: dict) -> np.ndarray:
    """
    Cria uma matriz de adjacência binária onde adj[i, j] = 1 se os nós i e j
    (na ordem ordenada das chaves) pertencem ao mesmo cluster (i != j).

    Parâmetros
    ----------
    cluster_dict : dict
        {no: cluster_id}

    Retorna
    -------
    np.ndarray
        Matriz N×N com 0/1.
    """
    nos = sorted(cluster_dict.keys())
    n = len(nos)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and cluster_dict[nos[i]] == cluster_dict[nos[j]]:
                adj[i, j] = 1
    return adj


def similarity_to_knn_adjacency(sim_matrix, k: int = 5) -> pd.DataFrame:
    """
    Convert similarity to a directed k-NN adjacency (0/1).

    Parameters
    ----------
    sim_matrix : np.ndarray or pd.DataFrame
        N×N similarity matrix.
    k : int
        Number of most similar neighbors (excluding self).

    Returns
    -------
    pd.DataFrame
        N×N directed adjacency (rows select their top-k columns).
    """
    if isinstance(sim_matrix, pd.DataFrame):
        labels = sim_matrix.index
        sim = sim_matrix.values
    else:
        sim = np.asarray(sim_matrix)
        labels = np.arange(sim.shape[0])

    if sim.shape[0] != sim.shape[1]:
        raise ValueError("sim_matrix must be square (N×N).")

    n = sim.shape[0]
    adj = np.zeros((n, n), dtype=int)

    for i in range(n):
        order = np.argsort(sim[i])[::-1]
        top_k = [j for j in order if j != i][:k]
        adj[i, top_k] = 1

    return pd.DataFrame(adj, index=labels, columns=labels)
