# models/bayesian_hmm/__init__.py

# Import the original BayesianHMM class
from .base import BayesianHMM

# Import the new BayesianRegimePersistence class
from .regime_persistence import BayesianRegimePersistence

# Export both classes
__all__ = ['BayesianHMM', 'BayesianRegimePersistence']