import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, Iterable, Tuple

from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LinearRegression
from scipy.stats import hmean 

import antropy as ant
import pycatch22

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
    x = np.asarray(x).ravel()
    n = x.size
    if n < 5:
        return {'arch_acf': np.nan, 'garch_acf': np.nan, 'arch_r2': np.nan, 'garch_r2': np.nan}

    order_ar = max(1, min(n - 1, int(np.floor(10 * np.log10(n)))))

    try:
        x_resid = AutoReg(x, lags=order_ar, trend='c').fit().resid
    except Exception:
        try:
            x_resid = AutoReg(x, lags=order_ar, trend='n').fit().resid
        except Exception:
            return {'arch_acf': np.nan, 'garch_acf': np.nan, 'arch_r2': np.nan, 'garch_r2': np.nan}

    max_lags = int(min(12, max(1, n - 1)))
    arch_acf_val = float(np.sum(acf(x_resid**2, nlags=max_lags, fft=False)[1:] ** 2))

    m = int(max(1, min(freq, n - 1)))
    lb_arch = acorr_ljungbox(x_resid, lags=[m], return_df=True)['lb_stat'].iloc[0]

    try:
        garch_fit = arch_model(x_resid, vol='GARCH', p=1, o=0, q=1, rescale=False).fit(disp='off')
        std_resid = garch_fit.std_resid
        garch_acf_val = float(np.sum(acf(std_resid**2, nlags=max_lags, fft=False)[1:] ** 2))
        lb_garch = acorr_ljungbox(std_resid, lags=[m], return_df=True)['lb_stat'].iloc[0]
    except Exception:
        garch_acf_val = np.nan
        lb_garch = np.nan

    return {'arch_acf': arch_acf_val, 'garch_acf': garch_acf_val,
            'arch_r2': float(lb_arch), 'garch_r2': float(lb_garch)}

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
    GE = nx.global_efficiency(G)
    if GE == 0:
        return 0.0, [0.0 for _ in G.nodes()]

    vulnerabilities = []
    for node in G.nodes():
        G_copy = G.copy()
        G_copy.remove_node(node)
        GE_i = nx.global_efficiency(G_copy)
        NV_i = (GE - GE_i) / GE
        vulnerabilities.append(NV_i)

    return max(vulnerabilities) if vulnerabilities else 0.0, vulnerabilities


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
    """
    A = np.asarray(A)
    mf: Dict[str, float] = {}

    # Per-node minimal positive edge cost
    row_mins = []
    for row in A:
        nz = row[row > 0]
        row_mins.append(np.mean(nz) if nz.size else 0.0)
    V_cost = row_mins

    g = nx.from_numpy_array(A)  # reads weights from array
    g = g.to_undirected()

    mf['min_V_cost']   = float(np.min(V_cost)) if len(V_cost) else np.nan
    mf['max_V_cost']   = float(np.max(V_cost)) if len(V_cost) else np.nan
    mf['avg_V_cost']   = float(np.mean(V_cost)) if len(V_cost) else np.nan
    mf['std_V_cost']   = float(np.std(V_cost)) if len(V_cost) else np.nan
    mf['median_V_cost']= float(np.median(V_cost)) if len(V_cost) else np.nan

    mf['nn_sum'] = float(np.sum([np.min(row[row > 0]) if np.any(row > 0) else 0.0 for row in A]))

    nz_edges = A[A > 0]
    if nz_edges.size:
        mf['min_edge_cost'] = float(np.min(nz_edges))
        mf['avg_edge_cost'] = float(np.mean(nz_edges))
        mf['std_edge_cost'] = float(np.std(nz_edges))
        mf['median_edge_cost'] = float(np.median(nz_edges))
    else:
        mf['min_edge_cost'] = mf['avg_edge_cost'] = mf['std_edge_cost'] = mf['median_edge_cost'] = np.nan

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
    mask = ~np.isnan(mat) & (mat > 0)
    mf['avg_geodesic_dist'] = float(np.mean(mat[mask])) if np.any(mask) else np.nan
    mf['harmonic_geodesic_dist'] = float(hmean(mat[mask])) if np.any(mask) else np.nan

    NV, _ = network_vulnerability(g)
    mf['network_vulnerability'] = float(NV)

    mf['clustering_coef4transitivity'] = float(nx.transitivity(g))
    ccw = nx.clustering(g, weight='weight')
    degree_sequence = [d for _, d in g.degree()]
    mf['max_degree'] = float(np.max(degree_sequence)) if degree_sequence else 0.0
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
    x = np.asarray(x).ravel()
    out: Dict[str, float] = {}

    # Catch22
    try:
        c22 = pycatch22.catch22_all(x)
        out.update({n: float(v) for n, v in zip(c22['names'], c22['values'])})
    except Exception:
        # populate with NaNs if Catch22 fails
        try:
            for name in pycatch22.catch22_all(np.array([0, 1])).get('names', []):
                out[name] = np.nan
        except Exception:
            pass  # package missing; skip

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
        out['SpectralEntropy']           = float(ant.spectral_entropy(x, sf=1.0, normalize=True))
        out['SVDEntropy']                = float(ant.svd_entropy(x))
    except Exception:
        # if antropy fails, leave those keys out; or set to NaN if preferred
        pass

    return out
