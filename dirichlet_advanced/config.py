# config.py
"""
Configuration settings for the volatility regime analysis.
"""
DATA_PATH = r"C:\Users\cinco\Desktop\Cinco-Quant\00_raw_data\testy1" # Example Path

# Default analysis parameters
DEFAULT_VOLATILITY_WINDOW = 20 # Used by simple rolling and some fallbacks
DEFAULT_NUM_REGIMES = 3
DEFAULT_GMM_RANDOM_STATE_BASE = 42

# Volatility Model Choice and Parameters
# Options: 'simple', 'ewma', 'garch', 'sv', 'nn', 'hybrid'
DEFAULT_VOLATILITY_METHOD = 'hybrid'

DEFAULT_GARCH_P = 1
DEFAULT_GARCH_Q = 1
DEFAULT_EWMA_DECAY = 0.94
DEFAULT_NN_LOOKBACK = 20 # For LSTM sequence length

# Bayesian Transition Analysis Parameters
# prior_type: 'jeffrey', 'uniform', 'empirical'
DEFAULT_PRIOR_TYPE = 'jeffrey'
DEFAULT_MIN_PRIOR = 0.1 # Min strength for empirical or adaptive prior
DEFAULT_MAX_PRIOR = 2.0 # Max strength for empirical or adaptive prior

# General Date Parameters
DEFAULT_START_DATE = '2010-01-01'
DEFAULT_END_DATE = None # None means use latest data available

# Output directory and file naming (ensure these are updated if you changed versioning)
EXCEL_OUTPUT_DIR_BASE = "volatility_analysis_excel_output_v4_advanced"
INTERMEDIATE_RESULTS_SUBDIR = "intermediate_calc_results_v4"
OVERALL_SUMMARY_FILENAME = 'overall_analysis_summary_v4.xlsx'

# Old prior strength, deprecated but might be in older task tuples if re-running
# This isn't actively used by new functions but helps process_worker handle old arg structures.
DEFAULT_PRIOR_STRENGTH = 1.0


# Ensure openpyxl is installed for Excel operations: pip install openpyxl
# Additional major dependencies:
# pip install arch scikit-learn torch torchvision torchaudio
# For PyMC3 (if attempting, highly dependent on Python version):
# pip install pymc3==3.11.5 "theano-pymc>=1.1.2"
# (Note: PyMC3 is tricky on Python 3.10+. Consider `pymc` for newer Pythons.)