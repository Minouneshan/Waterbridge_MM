"""Lightweight Bayesian belief-network implementation used by analysis.py.
This module keeps dependencies minimal and focuses on reproducibility: all
calculations are transparent, and the entire network is serialised via a simple
dictionary so results can be regenerated on any system.
"""
from __future__ import annotations

from typing import Dict, Tuple, List
import os

import matplotlib.pyplot as plt
import networkx as nx


class BayesianForecastModel:
    """A minimal Bayesian belief-network skeleton.

    Each forecast node is represented as ``{'prior_prob': p0, 'current_prob': p}``.
    Evidence updates are propagated along directed edges with a constant damping
    factor (0.5) – this is deliberately simple because detailed conditional
    probability tables are outside scope for the public repository.
    """

    def __init__(self, priors: Dict[str, float]):
        # Validate probabilities
        for fid, p in priors.items():
            if not 0 <= p <= 1:
                raise ValueError(f"Prior for {fid} must be a probability (0–1) – got {p!r}")

        self.forecasts: Dict[str, Dict[str, float]] = {
            fid: {"prior_prob": p, "current_prob": p} for fid, p in priors.items()
        }

        # Directed graph encoding causal dependencies
        self.graph: nx.DiGraph = nx.DiGraph()
        self.graph.add_nodes_from(priors.keys())

    # ------------------------------------------------------------------
    # Network construction helpers
    # ------------------------------------------------------------------
    def add_edge(self, source: str, target: str, weight: float = 0.5) -> None:
        if source not in self.graph or target not in self.graph:
            raise KeyError("Both nodes must exist in the graph before adding an edge.")
        self.graph.add_edge(source, target, weight=weight)

    # ------------------------------------------------------------------
    # Evidence integration – extremely simplified loopy belief propagation
    # ------------------------------------------------------------------
    def update_with_evidence(
        self,
        forecast_id: str,
        weight: float,
        description: str = "",
        source: str = "",
    ) -> None:
        """Apply a weighted evidence update to the focal forecast and propagate.

        Parameters
        ----------
        forecast_id : str
            Node receiving new evidence.
        weight : float
            Signed weight in log-odds space approximated by a linear probability
            bump here (for transparency). 0.8 = strong support, −0.8 = strong
            refutation. Must lie in −1..+1.
        description : str, optional
            Human-readable note stored for audit trail (not used in algorithm).
        source : str, optional
            Evidence provenance (e.g. "academic_research").
        """
        if not -1 <= weight <= 1:
            raise ValueError("Weight must be in [-1, +1].")
        if forecast_id not in self.forecasts:
            raise KeyError(f"Unknown forecast id: {forecast_id}")

        # Simple linear update in probability space (for demo purposes)
        p_old = self.forecasts[forecast_id]["current_prob"]
        p_new = max(0.0, min(1.0, p_old + weight * (1 - p_old))) if weight >= 0 else max(
            0.0, p_old + weight * p_old
        )
        self.forecasts[forecast_id]["current_prob"] = p_new

        # Propagate to children with damping factor
        damping = 0.5
        for child in self.graph.successors(forecast_id):
            pc_old = self.forecasts[child]["current_prob"]
            delta = damping * weight * (1 - pc_old) if weight >= 0 else damping * weight * pc_old
            self.forecasts[child]["current_prob"] = max(0.0, min(1.0, pc_old + delta))

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------
    def visualize_network(self, path: str) -> str:
        """Save a PNG rendering of the network with node colours = probability."""
        pos = nx.spring_layout(self.graph, seed=42)
        node_colors = [self.forecasts[n]["current_prob"] for n in self.graph.nodes]
        node_sizes = 2500

        plt.figure(figsize=(14, 10))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, arrows=True, arrowsize=15)
        nodes = nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            node_size=node_sizes,
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight="bold", font_color="white")
        cbar = plt.colorbar(nodes)
        cbar.set_label("Current probability", rotation=270, labelpad=15)
        plt.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path


# ----------------------------------------------------------------------
# Helper to instantiate the full Modern-Mercantilism network
# ----------------------------------------------------------------------

def create_comprehensive_model() -> BayesianForecastModel:
    """Return a belief network pre-populated with priors and edges."""
    priors = {
        "F1": 0.70,
        "F2": 0.80,
        "F3": 0.75,
        "F4": 0.65,
        "F5": 0.80,
        "F6": 0.75,
        "F7": 0.72,
        "F8": 0.65,
        "F9": 0.68,
        "F10": 0.70,
        "F11": 0.78,
        "F12": 0.77,
        "F13": 0.62,
        "F14": 0.75,
        "F15": 0.58,
        "F16": 0.60,
        "F17": 0.77,
        "F18": 0.72,
        "F19": 0.58,
        "F20": 0.78,
        "F21": 0.75,
        "F22": 0.68,
        "F23": 0.70,
        "F24": 0.80,
        "F25": 0.65,
    }

    model = BayesianForecastModel(priors)

    # Causal edges mirrored from code/simple_analysis.py & docs discussion
    edges: List[Tuple[str, str]] = [
        ("F1", "F3"),
        ("F1", "F6"),
        ("F6", "F7"),
        ("F23", "F14"),
        ("F14", "F24"),
        ("F24", "F13"),
        ("F22", "F10"),
        ("F10", "F17"),
        ("F17", "F18"),
        ("F25", "F12"),
        ("F12", "F20"),
        ("F20", "F21"),
        # Cross-cluster connections
        ("F1", "F25"),
        ("F25", "F21"),
        ("F12", "F13"),
    ]

    for src, dst in edges:
        model.add_edge(src, dst)

    return model
