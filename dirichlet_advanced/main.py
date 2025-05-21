# main.py
"""
Main script for volatility regime analysis, now with selectable advanced models.
"""
import os
import multiprocessing
from datetime import datetime
import shutil

import config # For default parameters and paths
from data_loader import load_stock_data
from process_workers import process_stock_wrapper, save_stock_excel_report_wrapper
from summary_generator import generate_overall_summary

# Helper to suppress TensorFlow/PyMC3/arch INFO/WARNING messages for cleaner output
def suppress_library_logs():
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # For TensorFlow if it were used
        import logging
        # Suppress PyMC3 INFO messages
        pm_logger = logging.getLogger('pymc3')
        if pm_logger: pm_logger.setLevel(logging.ERROR) # or logging.WARNING
        # Suppress arch INFO messages
        arch_logger = logging.getLogger('arch')
        if arch_logger: arch_logger.setLevel(logging.WARNING) # GARCH warnings
        # Suppress specific PyTorch warnings if needed, though less common for basic use
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch") # Example
        # Suppress Matplotlib font manager warnings (can be verbose)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    except Exception as e:
        print(f"Note: Could not suppress all library logs: {e}")


def run_gmm_analysis(
                   folder_path: str = config.DATA_PATH,
                   window: int = config.DEFAULT_VOLATILITY_WINDOW,
                   num_regimes_target: int = config.DEFAULT_NUM_REGIMES,
                   random_state_gmm_base: int = config.DEFAULT_GMM_RANDOM_STATE_BASE,
                   volatility_method: str = config.DEFAULT_VOLATILITY_METHOD,
                   garch_p: int = config.DEFAULT_GARCH_P, garch_q: int = config.DEFAULT_GARCH_Q,
                   ewma_decay: float = config.DEFAULT_EWMA_DECAY,
                   nn_lookback: int = config.DEFAULT_NN_LOOKBACK,
                   prior_type: str = config.DEFAULT_PRIOR_TYPE,
                   min_prior: float = config.DEFAULT_MIN_PRIOR,
                   max_prior: float = config.DEFAULT_MAX_PRIOR,
                   save_dir_base: str = config.EXCEL_OUTPUT_DIR_BASE,
                   start_date: str = config.DEFAULT_START_DATE,
                   end_date: str = config.DEFAULT_END_DATE,
                   num_processes: int = None,
                   force_recompute: bool = False
                   ):
    suppress_library_logs()
    print(f"Loading stock data from: {folder_path}")
    all_stock_data_loaded = load_stock_data(folder_path)
    if not all_stock_data_loaded: print("No stock data loaded. Exiting."); return {}, []
    print(f"Found {len(all_stock_data_loaded)} stock datasets.")

    eff_s_date = start_date if start_date else '1900-01-01'
    eff_e_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
    print(f"Analysis period: {eff_s_date} to {eff_e_date}. Vol method: {volatility_method}. Prior: {prior_type}.")

    os.makedirs(save_dir_base, exist_ok=True)
    intermediate_calc_dir = os.path.join(save_dir_base, config.INTERMEDIATE_RESULTS_SUBDIR)
    
    if force_recompute and os.path.exists(intermediate_calc_dir):
        print(f"Force recompute ON. Deleting: {intermediate_calc_dir}")
        try: shutil.rmtree(intermediate_calc_dir); print("Cache cleared.")
        except Exception as e: print(f"Error clearing cache {intermediate_calc_dir}: {e}")
    os.makedirs(intermediate_calc_dir, exist_ok=True)
    
    print("\n--- Starting Calculation Pass ---")
    tasks = []
    idx_counter = 0
    for symbol, df_orig in all_stock_data_loaded.items():
        if df_orig is None or df_orig.empty: continue
        current_rs_gmm = random_state_gmm_base + idx_counter
        # Ensure this tuple matches the expected structure in process_stock_wrapper
        task_args = (
            symbol, df_orig.copy(), window, num_regimes_target,
            None,  # Placeholder for the old/deprecated prior_strength
            eff_s_date, eff_e_date, intermediate_calc_dir, current_rs_gmm,
            volatility_method, garch_p, garch_q, ewma_decay, nn_lookback,
            prior_type, min_prior, max_prior
        )
        tasks.append(task_args)
        idx_counter += 1

    actual_procs = num_processes if num_processes else max(1, os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1)
    print(f"Using {actual_procs} processes for calculations.")
    
    calculated_results = {}
    if tasks:
        if actual_procs > 1 and len(tasks) > 1:
            # Using 'spawn' context for PyTorch/PyMC3 for better compatibility, esp. with CUDA
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=actual_procs) as pool:
                pool_res = pool.map(process_stock_wrapper, tasks)
        else:
            print("Sequential calculation.")
            pool_res = [process_stock_wrapper(task) for task in tasks]
        for sym, res_obj in pool_res:
            if res_obj: calculated_results[sym] = res_obj
    
    print(f"\n--- Calculation Complete. Results for {len(calculated_results)}/{len(tasks)} stocks. ---")
    if not calculated_results: return {}, []

    print("\n--- Starting Excel Report Generation ---")
    report_tasks = [(calculated_results[s], s, save_dir_base)
                    for s in sorted(calculated_results.keys()) if calculated_results[s]]
    if report_tasks:
        # Excel generation can be I/O bound or CPU bound (for plotting).
        # Using multiple processes can still help.
        excel_procs = min(actual_procs, 4) # Limit Excel processes to avoid overwhelming system if many stocks
        print(f"Using up to {excel_procs} processes for Excel reports.")
        if excel_procs > 1 and len(report_tasks) > 1:
            ctx = multiprocessing.get_context('spawn') # Use spawn for consistency
            with ctx.Pool(processes=excel_procs) as pool:
                pool.map(save_stock_excel_report_wrapper, report_tasks)
        else:
            print("Sequential Excel report generation.")
            for task in report_tasks: save_stock_excel_report_wrapper(task)
    print(f"\n--- Excel Reports Complete for {len(report_tasks)} stocks. ---")

    print("\n--- Starting Overall Summary Generation ---")
    summary_list = generate_overall_summary(calculated_results, num_regimes_target,
                                            save_dir_base, config.OVERALL_SUMMARY_FILENAME)
    print("\n--- Overall Summary Complete ---")
    return calculated_results, summary_list

if __name__ == "__main__":
    # Important for multiprocessing with PyTorch/PyMC3 on some systems (e.g. Windows)
    # Should be called only in the main execution block.
    # multiprocessing.freeze_support() # Might be needed on Windows if not using 'spawn'
    
    # For PyTorch + CUDA with multiprocessing, 'spawn' is often safer
    # This needs to be called before any Pool or Process is created.
    try:
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn': # Only set if not already spawn or if default needs override
            multiprocessing.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e: # Handles if it's already been set or other issues
        print(f"Could not set start method to 'spawn' (may be already set or on an OS where it's default/problematic): {e}. Using current method: {multiprocessing.get_start_method(allow_none=True)}")
    except Exception as e_set_method: # Catch any other exception during set_start_method
        print(f"An unexpected error occurred while trying to set start method: {e_set_method}")


    print("--- Initiating Advanced Volatility Regime Analysis (v4) ---")
    run_params = {
        "folder_path": config.DATA_PATH,
        "window": config.DEFAULT_VOLATILITY_WINDOW,
        "num_regimes_target": config.DEFAULT_NUM_REGIMES,
        "random_state_gmm_base": config.DEFAULT_GMM_RANDOM_STATE_BASE,
        "volatility_method": config.DEFAULT_VOLATILITY_METHOD, # 'hybrid' is good for testing GPU
        "garch_p": config.DEFAULT_GARCH_P, "garch_q": config.DEFAULT_GARCH_Q,
        "ewma_decay": config.DEFAULT_EWMA_DECAY,
        "nn_lookback": config.DEFAULT_NN_LOOKBACK,
        "prior_type": config.DEFAULT_PRIOR_TYPE,
        "min_prior": config.DEFAULT_MIN_PRIOR, "max_prior": config.DEFAULT_MAX_PRIOR,
        "save_dir_base": config.EXCEL_OUTPUT_DIR_BASE,
        "start_date": config.DEFAULT_START_DATE, "end_date": config.DEFAULT_END_DATE,
        "num_processes": None, # Uses default logic: os.cpu_count() - 1
        "force_recompute": False
    }
    # Example: Override for testing a specific, faster volatility method
    # run_params["volatility_method"] = "ewma"
    # run_params["force_recompute"] = True
    # run_params["num_processes"] = 1 # For easier debugging if issues arise

    print("Analysis Parameters:")
    for k, v_ in run_params.items(): print(f"- {k}: {v_ if v_ is not None else ('Default CPU' if k == 'num_processes' else 'Latest')}")
    
    final_results, final_summary = run_gmm_analysis(**run_params)
    
    print("\n--- Analysis Script Finished (v4) ---")
    if final_results: print(f"Processed data for {len(final_results)} stocks.")
    if final_summary: print(f"Summary entries for {len(final_summary)} stocks.")
    print(f"Outputs in: {os.path.abspath(run_params['save_dir_base'])}")