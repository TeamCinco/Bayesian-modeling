import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # No longer directly needed for plotting
import glob
import os
import math
import scipy.stats as stats
from datetime import datetime
from sklearn.mixture import GaussianMixture
from scipy.stats import norm # Ensure norm is imported
import pickle # For saving/loading intermediate results
import multiprocessing # For parallel processing
# Ensure openpyxl is installed: pip install openpyxl

# Set the path to your data
DATA_PATH = r"C:\Users\cinco\Desktop\Cinco-Quant\00_raw_data\5.16" # Example Path
# DATA_PATH = "/Users/jazzhashzzz/Desktop/Cinco-Quant/00_raw_data/5.15" # Example Path


def load_stock_data(folder_path):
    """Load all CSV files from a folder into a dictionary of dataframes."""
    all_data = {}
    for file_path in glob.glob(os.path.join(folder_path, "*.csv")):
        symbol = os.path.basename(file_path).split('.')[0]
        try:
            # print(f"Loading {symbol}...") # Can be verbose
            df = pd.read_csv(file_path)
            date_cols = [col for col in df.columns if any(date_str in col.lower()
                                                       for date_str in ['date', 'time'])]
            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                except Exception as e:
                    print(f"Could not parse dates in {symbol} ({date_col}): {e}, skipping date conversion")

            if 'close' in df.columns:
                all_data[symbol] = df
            else:
                close_cols = [col for col in df.columns if 'close' in col.lower()]
                if close_cols:
                    df.rename(columns={close_cols[0]: 'close'}, inplace=True)
                    all_data[symbol] = df
                    # print(f"Renamed column {close_cols[0]} to 'close' for {symbol}")
                else:
                    print(f"Skipping {symbol}: no 'close' column found")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_data

def calculate_volatility(df, window=20):
    log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

def count_transitions(regimes):
    regimes_clean = regimes.dropna()
    if regimes_clean.empty:
        return np.zeros((0, 0)) # Return empty array that can be reshaped if needed

    unique_regimes_from_data = sorted(regimes_clean.unique().astype(int))
    if not unique_regimes_from_data:
        return np.zeros((0,0))

    num_regimes_found = max(unique_regimes_from_data) + 1 if unique_regimes_from_data else 0
    if num_regimes_found == 0:
        return np.zeros((0,0))

    transition_counts = np.zeros((num_regimes_found, num_regimes_found))
    prev_regime = None
    for regime_val in regimes_clean:
        regime = int(regime_val)
        if prev_regime is not None:
            if 0 <= prev_regime < num_regimes_found and 0 <= regime < num_regimes_found:
                 transition_counts[prev_regime, regime] += 1
            # else: # Can be verbose
                # print(f"Warning: Regime out of bounds. Prev: {prev_regime}, Curr: {regime}, Max_idx: {num_regimes_found-1}")
        prev_regime = regime
    return transition_counts

def filter_data_by_date(data, start_date='2021-01-01', end_date=None):
    if not isinstance(data.index, pd.DatetimeIndex):
        # print("Warning: Data index is not a DatetimeIndex, skipping date filtering")
        return data
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize()
    filtered_data = data.loc[(data.index >= start) & (data.index <= end)].copy()
    return filtered_data

def bayesian_transition_analysis(transition_counts, prior_strength=1.0):
    num_regimes = transition_counts.shape[0]
    if num_regimes == 0: # Handles empty transition_counts
        return np.array([]), {}, {}

    posterior_samples = {}
    mean_probs = np.zeros_like(transition_counts, dtype=float)
    for from_regime in range(num_regimes):
        prior_alphas = np.ones(num_regimes) * prior_strength
        posterior_alphas = prior_alphas + transition_counts[from_regime]
        if posterior_alphas.sum() == 0:
            samples = np.zeros((10000, num_regimes))
            mean_probs[from_regime] = np.zeros(num_regimes)
        else:
            samples = np.random.default_rng().dirichlet(posterior_alphas, size=10000)
            mean_probs[from_regime] = posterior_alphas / posterior_alphas.sum()
        posterior_samples[from_regime] = samples
    return mean_probs, posterior_samples

def calculate_probability_intervals(samples, confidence=0.95):
    lower_bound = (1 - confidence) / 2
    upper_bound = 1 - lower_bound
    lower_probs = {}
    upper_probs = {}
    for from_regime, regime_samples in samples.items():
        lower_probs[from_regime] = np.quantile(regime_samples, lower_bound, axis=0)
        upper_probs[from_regime] = np.quantile(regime_samples, upper_bound, axis=0)
    return lower_probs, upper_probs

def identify_volatility_regimes_gmm(volatility, num_regimes_target=3, random_state=42):
    clean_vol = volatility.dropna()
    if len(clean_vol) < num_regimes_target: # GMM needs enough samples per component
        print(f"Warning: Not enough clean volatility data points ({len(clean_vol)}) for GMM with {num_regimes_target} regimes. Skipping GMM.")
        regimes = pd.Series(np.nan, index=volatility.index)
        regime_info = {'model': None, 'mapping': {}, 'means': [], 'variances': [], 'weights': []}
        return regimes, regime_info

    X = clean_vol.values.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=num_regimes_target, covariance_type='full',
        random_state=random_state, n_init=10
    )
    try:
        gmm.fit(X)
    except ValueError as e:
        print(f"GMM fitting error: {e}. Returning empty regimes.")
        regimes = pd.Series(np.nan, index=volatility.index)
        regime_info = {'model': None, 'mapping': {}, 'means': [], 'variances': [], 'weights': []}
        return regimes, regime_info

    regime_labels = gmm.predict(X)
    regimes_clean = pd.Series(regime_labels, index=clean_vol.index)

    gmm_means_flat = gmm.means_.flatten()
    regime_order = np.argsort(gmm_means_flat)
    regime_mapping = {old_label: new_label for new_label, old_label in enumerate(regime_order)}

    regimes_clean = regimes_clean.map(regime_mapping)
    regimes = pd.Series(np.nan, index=volatility.index)
    regimes.loc[regimes_clean.index] = regimes_clean

    # gmm.n_components is the actual number of components fitted
    actual_fitted_regimes = gmm.n_components
    regime_info = {
        'model': gmm,
        'mapping': regime_mapping,
        'means': [gmm.means_[regime_order[i]][0] for i in range(actual_fitted_regimes)],
        'variances': [gmm.covariances_[regime_order[i]][0][0] for i in range(actual_fitted_regimes)],
        'weights': [gmm.weights_[regime_order[i]] for i in range(actual_fitted_regimes)]
    }
    return regimes, regime_info


def analyze_stock_with_gmm(symbol, data, window=20, num_regimes_target=3, prior_strength=1.0,
                          start_date='2021-01-01', end_date='2025-05-14'):
    data = filter_data_by_date(data, start_date, end_date)
    if data.empty:
        print(f"No data for {symbol} in period {start_date}-{end_date}.")
        return None

    volatility = calculate_volatility(data, window)
    if volatility.dropna().empty:
        print(f"Not enough data for {symbol} to calculate volatility.")
        return None

    regimes, regime_info = identify_volatility_regimes_gmm(volatility, num_regimes_target)

    if not regime_info['means']: # GMM failed or no components found
        print(f"GMM could not identify regimes for {symbol}.")
        return None
    
    actual_num_regimes_identified_by_gmm = len(regime_info['means'])

    transition_counts = count_transitions(regimes)

    # Ensure transition_counts matrix matches the number of GMM identified regimes
    if transition_counts.shape[0] != actual_num_regimes_identified_by_gmm:
        new_tc = np.zeros((actual_num_regimes_identified_by_gmm, actual_num_regimes_identified_by_gmm))
        if transition_counts.ndim == 2 and transition_counts.shape[0] > 0 : # if some transitions were counted
            min_dim0 = min(transition_counts.shape[0], actual_num_regimes_identified_by_gmm)
            min_dim1 = min(transition_counts.shape[1], actual_num_regimes_identified_by_gmm)
            new_tc[:min_dim0, :min_dim1] = transition_counts[:min_dim0, :min_dim1]
        transition_counts = new_tc
    
    if actual_num_regimes_identified_by_gmm == 0: # If after all GMM had no regimes
        print(f"No GMM regimes for {symbol} to analyze transitions.")
        return None


    mean_probs, posterior_samples = bayesian_transition_analysis(transition_counts, prior_strength)
    if not posterior_samples and actual_num_regimes_identified_by_gmm > 0: # Check if posterior_samples is empty
        print(f"Bayesian analysis failed for {symbol} (posterior_samples empty).")
        # Provide default empty structures if needed, or return None
        # For robust downstream processing, let's ensure mean_probs matches expected GMM regime count
        if mean_probs.shape[0] != actual_num_regimes_identified_by_gmm:
            mean_probs = np.full((actual_num_regimes_identified_by_gmm, actual_num_regimes_identified_by_gmm), np.nan)
        # Create empty dicts for lower/upper probs
        lower_probs = {i: [np.nan]*actual_num_regimes_identified_by_gmm for i in range(actual_num_regimes_identified_by_gmm)}
        upper_probs = {i: [np.nan]*actual_num_regimes_identified_by_gmm for i in range(actual_num_regimes_identified_by_gmm)}
    elif actual_num_regimes_identified_by_gmm == 0: # GMM found no regimes
        mean_probs = np.array([])
        lower_probs, upper_probs = {}, {}
    else:
        lower_probs, upper_probs = calculate_probability_intervals(posterior_samples)


    current_regime = None
    cleaned_regimes = regimes.dropna()
    if not cleaned_regimes.empty:
        current_regime = int(cleaned_regimes.iloc[-1])
        # Validate current_regime against number of identified regimes
        if not (0 <= current_regime < actual_num_regimes_identified_by_gmm):
            print(f"Warning: Current regime {current_regime} for {symbol} is out of bounds for GMM identified regimes ({actual_num_regimes_identified_by_gmm}). Setting to None.")
            current_regime = None # Or handle as an error
    
    if current_regime is None and actual_num_regimes_identified_by_gmm > 0: # If we expected a current regime but couldn't get one
         print(f"No valid current regime found for {symbol} despite GMM identifying regimes.")
         # Potentially set to a default or return None
         # For now, we'll allow it to proceed and it will result in NaNs for current regime stats


    results = {
        'symbol': symbol, 'volatility': volatility, 'regimes': regimes,
        'regime_info': regime_info, 'transition_counts': transition_counts,
        'transition_probabilities': mean_probs,
        # 'posterior_samples': posterior_samples, # Usually large, can omit from final stored results if not directly used later
        'lower_probabilities': lower_probs, 'upper_probabilities': upper_probs,
        'current_regime': current_regime,
        'actual_num_regimes': actual_num_regimes_identified_by_gmm # Store this important piece of info
    }
    return results

def save_stock_excel_report(results, symbol, save_dir):
    """Saves the analysis results for a single stock to an Excel file with two sheets."""
    if results is None:
        print(f"No results to save for {symbol}.")
        return

    output_path = os.path.join(save_dir, f"{symbol}_analysis_results.xlsx")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # --- Sheet 1: Regime_Parameters ---
            regime_info = results.get('regime_info', {})
            num_actual_regimes = results.get('actual_num_regimes', 0)

            regime_data = []
            if num_actual_regimes > 0 and regime_info.get('means'):
                for i in range(num_actual_regimes):
                    mean_vol = regime_info['means'][i] if i < len(regime_info['means']) else np.nan
                    variance = regime_info['variances'][i] if i < len(regime_info['variances']) else np.nan
                    std_dev_vol = np.sqrt(variance) if not pd.isna(variance) and variance >=0 else np.nan
                    weight = regime_info['weights'][i] if i < len(regime_info['weights']) else np.nan
                    regime_data.append({
                        'Regime': i,
                        'Mean_Volatility': mean_vol,
                        'StdDev_Volatility': std_dev_vol,
                        'GMM_Weight': weight
                    })
            
            df_regime_params = pd.DataFrame(regime_data)
            if not df_regime_params.empty:
                df_regime_params.to_excel(writer, sheet_name='Regime_Parameters', index=False)
            else: # Write an empty placeholder or a message
                pd.DataFrame([{"Status": "No regime parameters identified"}]).to_excel(writer, sheet_name='Regime_Parameters', index=False)


            # --- Sheet 2: Current_Regime_Transitions ---
            current_regime = results.get('current_regime')
            transition_probs_matrix = results.get('transition_probabilities')
            lower_probs_dict = results.get('lower_probabilities', {})
            upper_probs_dict = results.get('upper_probabilities', {})
            
            transition_data = []
            if current_regime is not None and \
               transition_probs_matrix is not None and transition_probs_matrix.ndim == 2 and \
               0 <= current_regime < transition_probs_matrix.shape[0]:
                
                current_probs_row = transition_probs_matrix[current_regime]
                current_lower_row = lower_probs_dict.get(current_regime, [np.nan] * num_actual_regimes)
                current_upper_row = upper_probs_dict.get(current_regime, [np.nan] * num_actual_regimes)

                for to_regime in range(num_actual_regimes):
                    transition_data.append({
                        'From_Regime': current_regime,
                        'To_Regime': to_regime,
                        'Mean_Probability': current_probs_row[to_regime] if to_regime < len(current_probs_row) else np.nan,
                        '95%_CI_Lower': current_lower_row[to_regime] if to_regime < len(current_lower_row) else np.nan,
                        '95%_CI_Upper': current_upper_row[to_regime] if to_regime < len(current_upper_row) else np.nan,
                    })
            
            df_transitions = pd.DataFrame(transition_data)
            if not df_transitions.empty:
                 # Add a header row for current regime
                df_header = pd.DataFrame([{"Current_Regime_Analyzed": current_regime if current_regime is not None else "N/A"}])
                df_header.to_excel(writer, sheet_name='Current_Regime_Transitions', index=False, header=True, startrow=0)
                df_transitions.to_excel(writer, sheet_name='Current_Regime_Transitions', index=False, startrow=2) # Start data after header
            else:
                pd.DataFrame([{"Status": "No transition probabilities from current regime available or current regime not identified."}]).to_excel(writer, sheet_name='Current_Regime_Transitions', index=False)

        # print(f"Saved Excel report for {symbol} to {output_path}") # Can be verbose
    except Exception as e:
        print(f"Error saving Excel report for {symbol}: {e}")


# Worker function for multiprocessing
def process_stock_wrapper(args_tuple):
    symbol, data_df, window, num_regimes_target, prior_strength, start_date, end_date, intermediate_save_dir = args_tuple
    # Filename includes key parameters to distinguish cache for different settings
    params_str = f"w{window}_r{num_regimes_target}_p{prior_strength}_sd{start_date.replace('-', '')}_ed{end_date.replace('-', '') if end_date else 'None'}"
    checkpoint_file = os.path.join(intermediate_save_dir, f"{symbol}_results_{params_str}.pkl")


    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                results = pickle.load(f)
            if results and results.get('symbol') == symbol: # Basic validation
                return symbol, results
        except Exception as e:
            print(f"Error loading cached results for {symbol} (params: {params_str}): {e}. Recomputing...")

    results = analyze_stock_with_gmm(symbol, data_df, window, num_regimes_target, prior_strength,
                                     start_date, end_date)
    if results:
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error saving intermediate results for {symbol} to {checkpoint_file}: {e}")
    return symbol, results


def run_gmm_analysis(folder_path=DATA_PATH, window=20, num_regimes_target=3, prior_strength=1.0,
                   save_dir="volatility_excel_reports", start_date='2021-01-01', end_date='2025-05-14',
                   num_processes=None, force_recompute=False):
    print(f"Loading stock data from {folder_path}...")
    all_stock_data_loaded = load_stock_data(folder_path)

    if not all_stock_data_loaded:
        print("No valid stock data found.")
        return {}, []

    print(f"Found {len(all_stock_data_loaded)} stock datasets.")
    print(f"Analysis period: {start_date} to {end_date if end_date else 'latest available'}")

    results_dir = save_dir
    os.makedirs(results_dir, exist_ok=True)
    intermediate_results_dir = os.path.join(results_dir, "intermediate_calc_results")
    os.makedirs(intermediate_results_dir, exist_ok=True)

    if force_recompute:
        print("Force recompute is ON. Deleting existing intermediate pkl results...")
        for f_name in os.listdir(intermediate_results_dir):
            if f_name.endswith(".pkl"):
                try:
                    os.remove(os.path.join(intermediate_results_dir, f_name))
                except Exception as e:
                    print(f"Error deleting {f_name}: {e}")
    
    print("\n--- Starting Calculation Pass ---")
    tasks = []
    for symbol, data_df_orig in all_stock_data_loaded.items():
        tasks.append((symbol, data_df_orig.copy(), window, num_regimes_target, prior_strength,
                      start_date, end_date if end_date else datetime.now().strftime('%Y-%m-%d'), 
                      intermediate_results_dir))

    all_calculated_results = {}
    if not num_processes:
        num_processes = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
    print(f"Using {num_processes} processes for analysis.")

    if tasks:
        if num_processes > 1 and len(tasks) > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool_results = pool.map(process_stock_wrapper, tasks)
        else:
            print("Running in sequential mode.")
            pool_results = [process_stock_wrapper(task) for task in tasks]
        
        for symbol_key, res_obj in pool_results:
            if res_obj:
                all_calculated_results[symbol_key] = res_obj
    
    print(f"\n--- Calculation Pass Complete. Got results for {len(all_calculated_results)} stocks. ---")

    if not all_calculated_results:
        print("No stocks were successfully analyzed. Exiting.")
        return {}, []

    print("\n--- Starting Excel Report Generation Pass ---")
    summary_data_list = []
    sorted_symbols = sorted(all_calculated_results.keys())

    for symbol in sorted_symbols:
        results = all_calculated_results[symbol]
        if results is None:
            print(f"Skipping report for {symbol} due to no results from calculation pass.")
            continue
        
        save_stock_excel_report(results, symbol, results_dir) # Save individual Excel

        # Prepare overall summary entry
        current_regime = results.get('current_regime')
        actual_num_regimes = results.get('actual_num_regimes', 0)
        regime_info = results.get('regime_info', {})
        transition_probs_matrix = results.get('transition_probabilities')

        summary_entry = {'Symbol': symbol, 'Current_Regime': current_regime}
        
        # Current Volatility
        vol_series = results.get('volatility')
        if vol_series is not None and not vol_series.dropna().empty:
            summary_entry['Current_Volatility'] = vol_series.dropna().iloc[-1]
        else:
            summary_entry['Current_Volatility'] = np.nan

        # Mean Volatility of Current Regime
        if current_regime is not None and regime_info.get('means') and 0 <= current_regime < len(regime_info['means']):
            summary_entry['Regime_Mean_Volatility_Current'] = regime_info['means'][current_regime]
        else:
            summary_entry['Regime_Mean_Volatility_Current'] = np.nan

        # Transition probabilities from current regime
        if current_regime is not None and transition_probs_matrix is not None and \
           transition_probs_matrix.ndim == 2 and 0 <= current_regime < transition_probs_matrix.shape[0]:
            current_probs_row = transition_probs_matrix[current_regime]
            if 0 <= current_regime < len(current_probs_row):
                 summary_entry['Prob_Stay_Same_Regime'] = current_probs_row[current_regime]
            else:
                summary_entry['Prob_Stay_Same_Regime'] = np.nan

            for i in range(actual_num_regimes): # actual_num_regimes comes from results dict
                summary_entry[f'Prob_To_Regime_{i}'] = current_probs_row[i] if i < len(current_probs_row) else np.nan
        else: # Fill with NaNs if no current regime or probs
            summary_entry['Prob_Stay_Same_Regime'] = np.nan
            for i in range(num_regimes_target): # Use target as fallback for column names
                 summary_entry[f'Prob_To_Regime_{i}'] = np.nan


        # Parameters for all GMM regimes
        for i in range(actual_num_regimes):
            summary_entry[f'Regime_{i}_Mean_Vol'] = regime_info['means'][i] if regime_info.get('means') and i < len(regime_info['means']) else np.nan
            var = regime_info['variances'][i] if regime_info.get('variances') and i < len(regime_info['variances']) else np.nan
            summary_entry[f'Regime_{i}_StdDev_Vol'] = np.sqrt(var) if not pd.isna(var) and var >=0 else np.nan
            summary_entry[f'Regime_{i}_Weight'] = regime_info['weights'][i] if regime_info.get('weights') and i < len(regime_info['weights']) else np.nan
        
        summary_data_list.append(summary_entry)

    if summary_data_list:
        summary_df = pd.DataFrame(summary_data_list)
        if not summary_df.empty:
            summary_df.sort_values('Symbol', inplace=True)
            summary_path = os.path.join(results_dir, 'overall_analysis_summary.xlsx')
            try:
                summary_df.to_excel(summary_path, index=False, engine='openpyxl')
                print(f"\nOverall summary saved to {summary_path}")
            except Exception as e:
                print(f"Error saving overall summary Excel: {e}")
        else:
            print("Summary data list was populated but resulted in an empty DataFrame. No overall summary saved.")
    else:
        print("No summary data generated for the overall report.")

    print("\n--- Excel Report Generation Pass Complete ---")
    return all_calculated_results, summary_data_list


if __name__ == "__main__":
    # --- Configuration ---
    data_folder = DATA_PATH
    output_save_dir = "volatility_analysis_excel_output"

    window_size = 20
    num_target_regimes_for_gmm = 3 # GMM will try to find this many, might find fewer
    bayesian_prior_strength = 1.0
    analysis_start_date = '2010-01-01'
    analysis_end_date = '2025-05-15' # Use None for up to most recent data in CSVs

    num_parallel_processes = None # os.cpu_count() - 1 or specific number e.g. 4
    force_recomputation_on_rerun = False # Set to True to ignore cached .pkl files

    print("Bayesian Volatility Regime Analysis with GMM - Excel Output\n")
    print(f"Analysis Parameters:")
    print(f"- Data folder: {data_folder}")
    print(f"- Output directory: {output_save_dir}")
    print(f"- Volatility window: {window_size} days")
    print(f"- Target number of GMM regimes: {num_target_regimes_for_gmm}")
    print(f"- Prior strength for Bayesian transitions: {bayesian_prior_strength}")
    print(f"- Analysis period: {analysis_start_date} to {analysis_end_date if analysis_end_date else 'latest'}")
    print(f"- Number of parallel processes: {'Default (CPU count - 1)' if num_parallel_processes is None else num_parallel_processes}")
    print(f"- Force recomputation of intermediate files: {force_recomputation_on_rerun}")

    # On Windows, 'spawn' is often more stable for multiprocessing with complex objects.
    # if os.name == 'nt': 
    #    multiprocessing.set_start_method('spawn', force=True)

    results_dict, summary_list = run_gmm_analysis(
        folder_path=data_folder,
        window=window_size,
        num_regimes_target=num_target_regimes_for_gmm,
        prior_strength=bayesian_prior_strength,
        save_dir=output_save_dir,
        start_date=analysis_start_date,
        end_date=analysis_end_date,
        num_processes=num_parallel_processes,
        force_recompute=force_recomputation_on_rerun
    )

    print("\nAnalysis complete!")
    if results_dict:
        print(f"Successfully processed and generated data for {len(results_dict)} stocks.")
    if summary_list:
        print(f"Generated overall summary data for {len(summary_list)} stocks.")
    print(f"All Excel reports and summary saved in: {os.path.abspath(output_save_dir)}")