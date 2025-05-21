"""
Visualization package for Bayesian regime modeling.

Contains modules for visualizing regime detection results, 
creating interactive dashboards, and plotting various regime characteristics.
"""

from .regime_plots import (
    plot_regime_distribution,
    plot_regime_timeline,
    plot_regime_characteristics,
    plot_transition_heatmap,
    plot_performance_metrics
)

from .dashboard import create_dashboard

__all__ = [
    'plot_regime_distribution',
    'plot_regime_timeline',
    'plot_regime_characteristics',
    'plot_transition_heatmap',
    'plot_performance_metrics',
    'create_dashboard',
]
