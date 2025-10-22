import pickle
from pathlib import Path
from typing import Iterable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd


class GraphLoader:
    """
    Utility for loading and preparing graphs from disk (GraphML/GPickle) or a built-in test graph.

    Workflow
    --------
    - Optionally reindex nodes to 0..N-1 for compact arrays.
    - Collect node attributes (excluding 'label' and 'y') into 'x' and their names into 'a_labels'.
    - Assign binary labels 'y' based on membership in a known label list (Disney example).

    Parameters
    ----------
    base_path : str or Path, default="../graphs"
        Base directory for stored graphs.
    """

    def __init__(self, base_path: Union[str, Path] = "../graphs"):
        self.base_path = Path(base_path)
        self.disney_labels = [
            "B00005T5YC",
            "B00006LPHB",
            "B00004R99B",
            "B00005T7HD",
            "B00004T2SJ",
            "B00004WL3E",
        ]

    @staticmethod
    def _reindex_nodes(G: nx.Graph) -> nx.Graph:
        """Reindex nodes to 0..N-1 in the current iteration order (attributes preserved)."""
        mapping = {node: i for i, node in enumerate(G.nodes)}
        return nx.relabel_nodes(G, mapping, copy=True)

    @staticmethod
    def _process_node_features(G: nx.Graph) -> None:
        """
        Pack node attributes (except 'label' and 'y') into:
          - node['x']        : list of values
          - node['a_labels'] : list of attribute names (same order as 'x')
        """
        for node in G.nodes:
            x, xlabels = [], []
            for feature, value in G.nodes[node].items():
                if feature not in ("label", "y"):
                    x.append(value)
                    xlabels.append(feature)
            G.nodes[node]["x"] = x
            G.nodes[node]["a_labels"] = xlabels

    def _assign_labels(self, G: nx.Graph) -> None:
        """
        Assign binary label 'y' using the node 'label' attribute:
        y = 1 if label in self.disney_labels else 0.
        Nodes without 'label' get y=0.
        """
        for node in G.nodes:
            lbl = G.nodes[node].get("label")
            G.nodes[node]["y"] = int(lbl in self.disney_labels) if lbl is not None else 0

    def load_disney(self) -> nx.Graph:
        """
        Load and process the Disney graph stored as GraphML under base_path/disney.graphml.
        Returns an undirected graph with:
          - nodes reindexed 0..N-1
          - node features packed into 'x'/'a_labels'
          - binary target 'y' from self.disney_labels
        """
        gml_path = self.base_path / "disney.graphml"
        G = nx.read_graphml(gml_path)
        G = self._reindex_nodes(G)
        self._process_node_features(G)
        self._assign_labels(G)
        return G

    @staticmethod
    def create_test_graph() -> nx.Graph:
        """
        Create a small 12-node undirected test graph.
        Nodes 10 and 11 are marked as anomalies (y=1), others y=0.
        """
        G = nx.Graph()
        for i in range(12):
            G.add_node(i, y=1 if i in (10, 11) else 0)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 4),
            (5, 6), (6, 7), (7, 8), (8, 9), (5, 9),
            (4, 5), (5, 10), (4, 11), (6, 9), (6, 8),
            (7, 5), (0, 2), (1, 3), (2, 4),
        ]
        G.add_edges_from(edges)
        return G

    def load_graph(self, name: Optional[str] = None, file_path: Optional[Union[str, Path]] = None) -> nx.Graph:
        """
        Load a graph by name ('test' | 'disney' | <filename without suffix>) or by explicit path.

        Parameters
        ----------
        name : str, optional
            'test' → built-in test graph
            'disney' → base_path/disney.graphml
            other → base_path/{name}.gpickle
        file_path : str or Path, optional
            Explicit path to a file. Supported: .gpickle (via NetworkX), .graphml

        Returns
        -------
        G : nx.Graph

        Notes
        -----
        - For untrusted inputs, prefer GraphML. Loading pickles is inherently unsafe.
        """
        if file_path is not None:
            path = Path(file_path)
            if path.suffix.lower() in (".gpickle", ".gpkl"):
                G = nx.read_gpickle(path)
            elif path.suffix.lower() in (".graphml", ".xml"):
                G = nx.read_graphml(path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            return G

        if name is None:
            raise ValueError("Either 'name' or 'file_path' must be provided.")

        if name == "test":
            return self.create_test_graph()
        if name == "disney":
            return self.load_disney()

        # default: try base_path/{name}.gpickle
        gp = self.base_path / f"{name}.gpickle"
        if not gp.exists():
            raise FileNotFoundError(f"Graph not found: {gp}")
        return nx.read_gpickle(gp)


class GraphLoader_TSAD:
    """
    Lightweight loader/adapter for Time Series Anomaly Detection contexts that start from an
    adjacency (similarity/distance) matrix rather than a pre-serialized graph.

    Parameters
    ----------
    G : nx.Graph, optional
        An existing graph to store alongside this loader (not required).
    """

    def __init__(self, G: Optional[nx.Graph] = None):
        self.G = G

    @staticmethod
    def _reindex_nodes(G: nx.Graph) -> nx.Graph:
        """Reindex nodes to 0..N-1 (attributes preserved)."""
        mapping = {node: i for i, node in enumerate(G.nodes)}
        return nx.relabel_nodes(G, mapping, copy=True)

    @staticmethod
    def _process_node_features(G: nx.Graph) -> None:
        """Collect node attributes (excluding 'label' and 'y') into 'x' and names into 'a_labels'."""
        for node in G.nodes:
            x, xlabels = [], []
            for feature, value in G.nodes[node].items():
                if feature not in ("label", "y"):
                    x.append(value)
                    xlabels.append(feature)
            G.nodes[node]["x"] = x
            G.nodes[node]["a_labels"] = xlabels

    @staticmethod
    def _assign_labels(G: nx.Graph) -> None:
        """Initialize all node labels 'y' to 0 (placeholder)."""
        for node in G.nodes:
            G.nodes[node]["y"] = 0

    def load(self, A: Union[pd.DataFrame, np.ndarray], undirected: bool = True) -> nx.Graph:
        """
        Build a graph from an adjacency matrix, then standardize node indices and attributes.

        Parameters
        ----------
        A : pd.DataFrame or np.ndarray
            Adjacency (weights allowed). If DataFrame, its index/columns are node labels.
        undirected : bool, default=True
            If True, convert to undirected with summed or mirrored weights.

        Returns
        -------
        G : nx.Graph
            Graph with:
              - nodes reindexed 0..N-1
              - 'x'/'a_labels' features packed (if any existed)
              - 'y' initialized to 0 for all nodes
        """
        if isinstance(A, pd.DataFrame):
            G = nx.from_pandas_adjacency(A)
        else:
            A = np.asarray(A)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError("A must be a square 2D array or DataFrame.")
            G = nx.from_numpy_array(A)

        if undirected and isinstance(G, nx.DiGraph):
            G = G.to_undirected()

        G = self._reindex_nodes(G)
        self._process_node_features(G)
        self._assign_labels(G)
        return G
