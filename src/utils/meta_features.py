import numpy as np
import networkx as nx
from typing import Dict

from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import hmean

import antropy as ant
import pycatch22

from .multiplex import shannon_entropy


_HETEROGENEITY_KEYS = ("arch_acf", "garch_acf", "arch_r2", "garch_r2")


def _empty_heterogeneity() -> Dict[str, float]:
    """Return the default dictionary filled with ``np.nan`` diagnostics."""

    return {key: np.nan for key in _HETEROGENEITY_KEYS}


try:  # pragma: no branch - lightweight helper evaluated once at import time
    _CATCH22_NAMES = tuple(pycatch22.catch22_all(np.array([0.0, 1.0]))["names"])
except Exception:  # pragma: no cover - fallback when pycatch22 misbehaves at import
    _CATCH22_NAMES = ()


def heterogeneity(x: np.ndarray, freq: int = 1) -> Dict[str, float]:
    """
    Quantify conditional heteroskedasticity via AR-whitening and GARCH.

    Steps
    -----
    1) Fit AutoReg to x → residuals.
    2) On residuals:
       - arch_acf : sum of squares of ACF of squared residuals (lags 1..L, L≤12)
       - arch_r2  : Ljung–Box lb_stat at lag=freq
    3) Fit GARCH(1,1) on residuals, repeat diagnostics on standardized resid:
       - garch_acf, garch_r2

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    freq : int, default=1
        Lag used for Ljung–Box.

    Returns
    -------
    dict with keys {'arch_acf','garch_acf','arch_r2','garch_r2'}.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 5:
        return _empty_heterogeneity()

    order_ar = max(1, min(n - 1, int(np.floor(10 * np.log10(n)))))
    x_resid = None
    for trend in ("c", "n"):
        try:
            x_resid = AutoReg(x, lags=order_ar, trend=trend).fit().resid
            break
        except Exception:
            continue

    if x_resid is None or x_resid.size == 0:
        return _empty_heterogeneity()

    max_lags = int(min(12, max(1, n - 1)))

    arch_acf_val = np.nan
    try:
        acf_vals = acf(x_resid ** 2, nlags=max_lags, fft=False)
        arch_acf_val = float(np.sum(acf_vals[1:] ** 2))
    except Exception:
        pass

    m = int(max(1, min(freq, n - 1)))
    lb_arch = np.nan
    try:
        lb_arch = float(acorr_ljungbox(x_resid, lags=[m], return_df=True)["lb_stat"].iloc[0])
    except Exception:
        pass

    garch_acf_val = np.nan
    lb_garch = np.nan
    try:
        garch_fit = arch_model(x_resid, vol="GARCH", p=1, o=0, q=1, rescale=False).fit(disp="off")
        std_resid = garch_fit.std_resid
        acf_vals = acf(std_resid ** 2, nlags=max_lags, fft=False)
        garch_acf_val = float(np.sum(acf_vals[1:] ** 2))
        lb_garch = float(acorr_ljungbox(std_resid, lags=[m], return_df=True)["lb_stat"].iloc[0])
    except Exception:
        pass

    return {
        "arch_acf": arch_acf_val,
        "garch_acf": garch_acf_val,
        "arch_r2": lb_arch,
        "garch_r2": lb_garch,
    }

def network_vulnerability(G: nx.Graph):
    """
    Network Vulnerability (NV) as the maximum relative drop in global efficiency (GE)
    upon removal of a single node:
        NV = max_i (GE - GE_i) / GE

    Returns
    -------
    NV : float
        Max vulnerability. Returns 0.0 if GE == 0.
    vulnerabilities : list[float]
        Per-node vulnerabilities in node iteration order.
    """
    nodes = list(G.nodes())
    if not nodes:
        return 0.0, []

    GE = nx.global_efficiency(G)
    if GE == 0:
        return 0.0, [0.0 for _ in nodes]

    vulnerabilities = []
    for node in nodes:
        subgraph = G.copy()
        subgraph.remove_node(node)
        GE_i = nx.global_efficiency(subgraph)
        NV_i = max((GE - GE_i) / GE, 0.0)
        vulnerabilities.append(NV_i)

    return (max(vulnerabilities) if vulnerabilities else 0.0), vulnerabilities


def _safe_reduce(values: np.ndarray, reducer) -> float:
    """Apply ``reducer`` to ``values`` returning ``np.nan`` on empty input."""

    if values.size == 0:
        return float(np.nan)
    return float(reducer(values))


def get_mf(A: np.ndarray, tau: int = 6) -> Dict[str, float]:
    """
    Compute multiplex-free (single-layer) graph metrics from an adjacency matrix A.

    Parameters
    ----------
    A : np.ndarray
        NxN adjacency (weighted allowed). Zeros treated as no-edge.
    tau : int, optional
        Unused placeholder kept for API compatibility.

    Returns
    -------
    dict
        A dictionary of scalar metrics (cost stats, distances, efficiency, clustering, entropy, etc.).

    Raises
    ------
    ValueError
        If ``A`` is not a square matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")

    mf: Dict[str, float] = {}

    pos_mask = A > 0
    row_has_pos = pos_mask.any(axis=1)
    row_sum = np.where(pos_mask, A, 0.0).sum(axis=1)
    pos_counts = pos_mask.sum(axis=1)
    row_mean_cost = np.divide(
        row_sum,
        pos_counts,
        out=np.zeros_like(row_sum, dtype=float),
        where=pos_counts > 0,
    )

    mf['min_V_cost'] = _safe_reduce(row_mean_cost, np.min)
    mf['max_V_cost'] = _safe_reduce(row_mean_cost, np.max)
    mf['avg_V_cost'] = _safe_reduce(row_mean_cost, np.mean)
    mf['std_V_cost'] = _safe_reduce(row_mean_cost, np.std)
    mf['median_V_cost'] = _safe_reduce(row_mean_cost, np.median)

    min_per_row = np.zeros(A.shape[0], dtype=float)
    if np.any(row_has_pos):
        positive_values = np.where(pos_mask, A, np.nan)
        with np.errstate(all='ignore'):
            min_per_row[row_has_pos] = np.nanmin(positive_values[row_has_pos], axis=1)
    mf['nn_sum'] = float(min_per_row.sum())

    nz_edges = A[pos_mask]
    if nz_edges.size:
        mf['min_edge_cost'] = float(np.min(nz_edges))
        mf['avg_edge_cost'] = float(np.mean(nz_edges))
        mf['std_edge_cost'] = float(np.std(nz_edges))
        mf['median_edge_cost'] = float(np.median(nz_edges))
    else:
        mf['min_edge_cost'] = mf['avg_edge_cost'] = mf['std_edge_cost'] = mf['median_edge_cost'] = np.nan

    g = nx.from_numpy_array(A, create_using=nx.Graph)

    GE = nx.global_efficiency(g)
    mf['global_efficiency'] = float(GE)

    # All-pairs shortest-path distances (weighted)
    dists = dict(nx.all_pairs_dijkstra_path_length(g, weight='weight'))
    # Build dense matrix with NaN where unreachable
    nodes = sorted(g.nodes())
    mat = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
    idx = {n:i for i, n in enumerate(nodes)}
    for u, du in dists.items():
        for v, dv in du.items():
            mat[idx[u], idx[v]] = dv
    # ignore zeros (diagonal) and NaNs
    mask = np.isfinite(mat) & (mat > 0)
    mf['avg_geodesic_dist'] = float(np.mean(mat[mask])) if np.any(mask) else np.nan
    mf['harmonic_geodesic_dist'] = float(hmean(mat[mask])) if np.any(mask) else np.nan

    NV, _ = network_vulnerability(g)
    mf['network_vulnerability'] = float(NV)

    mf['clustering_coef4transitivity'] = float(nx.transitivity(g))
    ccw = nx.clustering(g, weight='weight')
    degree_sequence = np.fromiter((d for _, d in g.degree()), dtype=float)
    mf['max_degree'] = float(np.max(degree_sequence)) if degree_sequence.size else 0.0
    mf['clustering_coefficient'] = float(np.mean(list(ccw.values()))) if ccw else 0.0
    mf['degree_distribution_entropy'] = float(shannon_entropy(g))

    return mf


def extractC22Antropy(x: np.ndarray, freq: int = 1) -> Dict[str, float]:
    """
    Compute Catch22 + AntroPy features on a 1D series.

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    freq : int, optional
        Unused here; placeholder for API symmetry.

    Returns
    -------
    dict
        Keys include 22 Catch22 features plus several AntroPy complexity/entropy metrics.

    Notes
    -----
    Requires `pycatch22` and `antropy` to be installed.
    """
    x = np.asarray(x, dtype=float).ravel()
    out: Dict[str, float] = {}

    # Catch22
    try:
        c22 = pycatch22.catch22_all(x)
        out.update({n: float(v) for n, v in zip(c22['names'], c22['values'])})
    except Exception:
        if _CATCH22_NAMES:
            out.update({name: np.nan for name in _CATCH22_NAMES})

    # AntroPy
    try:
        out['AppEntropy']                = float(ant.perm_entropy(x, normalize=True))
        out['DetrendedFluctuation']      = float(ant.detrended_fluctuation(x))
        out['HiguchiFractalDimension']   = float(ant.higuchi_fd(x))
        hj_p = ant.hjorth_params(x)
        out['HjorthComplexity']          = float(hj_p[1])
        out['HjorthMobility']            = float(hj_p[0])
        out['KatzFractalDimension']      = float(ant.katz_fd(x))
        out['PetrosianFractalDimension'] = float(ant.petrosian_fd(x))
        sf = float(max(freq, 1))
        out['SpectralEntropy']           = float(ant.spectral_entropy(x, sf=sf, normalize=True))
        out['SVDEntropy']                = float(ant.svd_entropy(x))
    except Exception:
        # if antropy fails, leave those keys out; or set to NaN if preferred
        pass

    return out
