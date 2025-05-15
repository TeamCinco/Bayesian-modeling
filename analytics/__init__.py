"""
Analytics package for Bayesian regime modeling.

Contains modules for extended statistical analysis of regimes,
transition analysis, and comprehensive regime characterization.
"""

from .regime_statistics import (
    calculate_extended_regime_statistics,
    calculate_regime_correlations,
    analyze_risk_adjusted_metrics
)

from .transition_analysis import (
    calculate_transition_probabilities,
    analyze_regime_duration,
    predict_regime_stability
)

__all__ = [
    'calculate_extended_regime_statistics',
    'calculate_regime_correlations',
    'analyze_risk_adjusted_metrics',
    'calculate_transition_probabilities',
    'analyze_regime_duration',
    'predict_regime_stability'
]
