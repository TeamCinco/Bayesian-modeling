# process_workers.py
"""
Wrapper functions for multiprocessing tasks. Updated for new parameters in checkpoint name.
"""
import os
import pickle
from stock_analyzer import analyze_stock_with_gmm
from excel_reporter import save_stock_excel_report_improved # Use the improved reporter
import config

def process_stock_wrapper(args_tuple: tuple) -> tuple:
    """
    Wrapper for `analyze_stock_with_gmm` for multiprocessing.
    Handles caching and passes all necessary parameters.
    """
    # Expected full tuple structure from main.py:
    # (symbol, data_df, window, num_r_target,
    #  None, # deprecated_prior_strength placeholder
    #  s_date, e_date, int_dir, rs_gmm,
    #  vol_method, garch_p, garch_q, ewma_decay, nn_lookback,
    #  prior_type, min_prior, max_prior)
    # Total 17 arguments

    try:
        symbol, data_df, window, num_r_target, \
        _deprecated_prior_strength, \
        s_date, e_date, int_dir, rs_gmm, \
        vol_method, garch_p, garch_q, ewma_decay, nn_lookback, \
        prior_type, min_prior, max_prior = args_tuple
    except ValueError as e:
        # This fallback might be hit if main.py doesn't send all 17 args.
        # Ensure main.py's task_args matches this expected structure.
        stock_sym_for_error = args_tuple[0] if args_tuple and len(args_tuple) > 0 else "Unknown"
        print(f"Error unpacking args_tuple for {stock_sym_for_error} (expected 17, got {len(args_tuple)}): {e}. Using defaults for missing advanced params.")
        # Fallback to a structure that might work with older pickled files or simpler calls
        # (Adjust this based on how you expect to call it during transitions)
        symbol, data_df, window, num_r_target, \
        _deprecated_prior_strength, \
        s_date, e_date, int_dir, rs_gmm = args_tuple[:9] # Base args
        # Defaults for advanced params
        vol_method = config.DEFAULT_VOLATILITY_METHOD
        garch_p, garch_q = config.DEFAULT_GARCH_P, config.DEFAULT_GARCH_Q
        ewma_decay = config.DEFAULT_EWMA_DECAY
        nn_lookback = config.DEFAULT_NN_LOOKBACK
        prior_type = config.DEFAULT_PRIOR_TYPE
        min_prior, max_prior = config.DEFAULT_MIN_PRIOR, config.DEFAULT_MAX_PRIOR


    sd_str = s_date.replace('-', '') if s_date else "None"
    ed_str = e_date.replace('-', '') if e_date else "None"
    
    short_vol_method = vol_method[:3] if isinstance(vol_method, str) else "unk"
    short_prior_type = prior_type[:3] if isinstance(prior_type, str) else "unk"

    params_str = (f"w{window}_r{num_r_target}_sd{sd_str}_ed{ed_str}_rs{rs_gmm}"
                  f"_vm{short_vol_method}" # e.g., vmhyb for hybrid
                  f"_pt{short_prior_type}"   # e.g., ptjef for jeffrey
                  f"_v4") # Version indicator for cache
    checkpoint_file = os.path.join(int_dir, f"{symbol}_results_{params_str}.pkl")

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f: results = pickle.load(f)
            if results and isinstance(results, dict) and results.get('symbol') == symbol:
                if 'probability_mass_functions' in results: # Key field from v4 results
                    return symbol, results
        except Exception as e_load:
            print(f"Error loading cached {symbol} (params: {params_str}): {e_load}. Recomputing...")

    if data_df is None or data_df.empty: return symbol, None

    results = analyze_stock_with_gmm(
        symbol=symbol, data=data_df,
        volatility_method=vol_method, window=window, num_regimes_target=num_r_target,
        garch_p=garch_p, garch_q=garch_q, ewma_decay=ewma_decay, nn_lookback=nn_lookback,
        prior_type=prior_type, min_prior=min_prior, max_prior=max_prior,
        start_date=s_date, end_date=e_date, random_state_gmm=rs_gmm
    )
    if results:
        try:
            os.makedirs(int_dir, exist_ok=True)
            with open(checkpoint_file, 'wb') as f: pickle.dump(results, f)
        except Exception as e_save:
            print(f"Error saving intermediate for {symbol} to {checkpoint_file}: {e_save}")
    return symbol, results

def save_stock_excel_report_wrapper(args_tuple: tuple) -> str:
    """Wrapper for the improved Excel report saving."""
    results_obj, symbol_key, report_save_dir = args_tuple
    if results_obj:
        save_stock_excel_report_improved(results_obj, symbol_key, report_save_dir)
    return symbol_key