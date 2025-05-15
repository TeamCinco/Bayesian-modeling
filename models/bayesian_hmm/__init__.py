# Import the main class
from .base import BayesianHMM

# Import key utility functions that might be useful at package level
from .math_utils import gaussian_pdf, log_sum_exp

__all__ = ['BayesianHMM', 'gaussian_pdf', 'log_sum_exp']
