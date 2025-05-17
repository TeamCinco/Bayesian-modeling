import pandas as pd
import numpy as np
import glob
import os
import math
# import scipy.stats as stats # No longer directly used, np and pd methods cover it
from datetime import datetime
from sklearn.mixture import GaussianMixture
# from scipy.stats import norm # norm from scipy.stats is not explicitly used here.
import pickle # For saving/loading intermediate results and backtest results
import multiprocessing # For parallel processing in run_gmm_analysis
# Ensure openpyxl is installed: pip install openpyxl

# Set the path to your data
DATA_PATH = r"C:\Users\cinco\Desktop\Cinco-Quant\00_raw_data\test" # Example Path

def load_stock_data(folder_path):
    """Load all CSV files from a folder into a dictionary of dataframes, standardizing OHLC column names."""
    all_data = {}
    name_map = {
        'open': ['open', 'o', 'first'],
        'high': ['high', 'h', 'max'],
        'low': ['low', 'l', 'min'],
        'close': ['close', 'c', 'last', 'price'] # Add 'price' as a common alternative for close
    }
    required_cols_std = ['open', 'high', 'low', 'close']

    for file_path in glob.glob(os.path.join(folder_path, "*.csv")):
        symbol = os.path.basename(file_path).split('.')[0]
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip() # Standardize column names

            rename_dict = {}
            found_cols = {}

            for std_name, potential_names in name_map.items():
                for pot_name in potential_names:
                    if pot_name in df.columns:
                        if std_name not in found_cols: # Take the first match
                           rename_dict[pot_name] = std_name
                           found_cols[std_name] = True
                        break
            
            df.rename(columns=rename_dict, inplace=True)

            # Ensure all standardized OHLC columns exist, coerce to numeric, handle missing
            for col_std in required_cols_std:
                if col_std not in df.columns:
                    df[col_std] = np.nan # Add missing OHLC columns as NaN
                df[col_std] = pd.to_numeric(df[col_std], errors='coerce')
            
            # Require at least 'close' to be mostly non-NaN
            if 'close' not in df.columns or df['close'].isnull().all():
                # print(f"Skipping {symbol}: 'close' column missing or all NaN after processing.")
                continue
            
            df.dropna(subset=['close'], inplace=True) # Drop rows where 'close' is NaN after coercion
            if df.empty:
                # print(f"Skipping {symbol}: DataFrame empty after dropping NaN 'close' values.")
                continue

            date_cols = [col for col in df.columns if any(date_str in col.lower()
                                                       for date_str in ['date', 'time', 'timestamp'])]
            if date_cols:
                date_col_name = date_cols[0] # Pick the first potential date column
                try:
                    # Attempt to convert to datetime, handling various potential formats
                    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
                    df.dropna(subset=[date_col_name], inplace=True) # Drop rows where date conversion failed
                    if df.empty: continue

                    df.set_index(date_col_name, inplace=True)
                    df.sort_index(inplace=True)
                except Exception as e:
                    print(f"Could not parse or set dates for {symbol} ({date_col_name}): {e}. Skipping stock.")
                    continue
            elif isinstance(df.index, pd.DatetimeIndex):
                 if not df.index.is_monotonic_increasing:
                    df.sort_index(inplace=True)
            else:
                # print(f"Warning: No date column found or index not DatetimeIndex for {symbol}. Analysis might be affected or skipped.")
                continue # Skip if no valid date index can be established

            all_data[symbol] = df[required_cols_std + [col for col in df.columns if col not in required_cols_std and col in found_cols.values()]].copy() # Keep only std OHLC + original other mapped cols
            all_data[symbol] = df # Keep all columns after renaming and date processing

        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")
    return all_data

def calculate_volatility(df, window=20):
    if 'close' not in df.columns or df['close'].isnull().all():
        return pd.Series(dtype=float, index=df.index if df is not None else None)
    
    prices = df['close'][df['close'] > 0].copy() 
    if prices.empty:
        return pd.Series(dtype=float, index=df.index if df is not None else None)
        
    shifted_prices = prices.shift(1)
    valid_for_log_return = (prices > 0) & (shifted_prices > 0) 
    log_returns = pd.Series(np.nan, index=df.index, dtype=float)
    
    if valid_for_log_return.any():
        computed_log_returns_subset = np.log(prices[valid_for_log_return] / shifted_prices[valid_for_log_return])
        log_returns.loc[computed_log_returns_subset.index] = computed_log_returns_subset

    if log_returns.dropna().empty:
        return pd.Series(dtype=float, index=df.index if df is not None else None)
        
    available_log_returns_count = len(log_returns.dropna())
    min_p = max(1, window // 2) # Ensure min_periods is at least 1
    if available_log_returns_count < min_p : # If not enough data for even min_periods
        min_p = max(1,available_log_returns_count)

    volatility = log_returns.rolling(window=window, min_periods=min_p).std() * np.sqrt(252)
    return volatility


def count_transitions(regimes):
    regimes_clean = regimes.dropna()
    if regimes_clean.empty:
        return np.zeros((0, 0))
    unique_regimes_from_data = sorted(regimes_clean.unique().astype(int))
    if not unique_regimes_from_data: # e.g. if all regimes were NaN and then dropped
        return np.zeros((0,0))
    # num_regimes_found should be based on max label + 1, assuming labels are 0-indexed
    num_regimes_found = max(unique_regimes_from_data) + 1 if unique_regimes_from_data else 0
    if num_regimes_found == 0:
        return np.zeros((0,0))

    transition_counts = np.zeros((num_regimes_found, num_regimes_found))
    prev_regime = None
    for regime_val in regimes_clean:
        regime = int(regime_val) # Ensure it's an int for indexing
        if prev_regime is not None:
            # Check bounds before assignment
            if 0 <= prev_regime < num_regimes_found and 0 <= regime < num_regimes_found:
                 transition_counts[prev_regime, regime] += 1
        prev_regime = regime
    return transition_counts

def filter_data_by_date(data, start_date_str='2010-01-01', end_date_str=None):
    if not isinstance(data.index, pd.DatetimeIndex):
        # print(f"Data for a stock does not have a DatetimeIndex. Skipping date filtering.")
        return data 
    if not data.index.is_monotonic_increasing: 
        data = data.sort_index()

    start_ts = pd.Timestamp(start_date_str)
    end_ts = pd.Timestamp(end_date_str) if end_date_str else pd.Timestamp.now().normalize() 

    if data.index.tz is not None:
        start_ts = start_ts.tz_localize(data.index.tz) if start_ts.tzinfo is None else start_ts.tz_convert(data.index.tz)
        end_ts = end_ts.tz_localize(data.index.tz) if end_ts.tzinfo is None else end_ts.tz_convert(data.index.tz)
    else: # data.index is tz-naive
        start_ts = start_ts.tz_localize(None) if start_ts.tzinfo is not None else start_ts
        end_ts = end_ts.tz_localize(None) if end_ts.tzinfo is not None else end_ts
            
    filtered_data = data.loc[(data.index >= start_ts) & (data.index <= end_ts)].copy()
    return filtered_data

def bayesian_transition_analysis(transition_counts, prior_strength=1.0):
    num_regimes = transition_counts.shape[0]
    if num_regimes == 0:
        return np.array([]), {} 
    posterior_samples = {}
    mean_probs = np.zeros_like(transition_counts, dtype=float)
    
    rng = np.random.default_rng() # Create one RNG instance

    for from_regime in range(num_regimes):
        prior_alphas = np.ones(num_regimes) * prior_strength
        posterior_alphas = prior_alphas + transition_counts[from_regime]

        if np.all(posterior_alphas <= 0) or posterior_alphas.sum() == 0 : # Summing non-positives, or all zeros
            if num_regimes > 0:
                samples = np.full((10000, num_regimes), 1.0/num_regimes)
                mean_probs[from_regime, :] = 1.0/num_regimes
            else: 
                samples = np.empty((10000, 0)) 
        else:
            safe_posterior_alphas = np.maximum(posterior_alphas, 1e-9) 
            samples = rng.dirichlet(safe_posterior_alphas, size=10000)
            mean_probs[from_regime] = safe_posterior_alphas / safe_posterior_alphas.sum()
        posterior_samples[from_regime] = samples
    return mean_probs, posterior_samples

def calculate_probability_intervals(samples, confidence=0.95):
    lower_bound = (1 - confidence) / 2
    upper_bound = 1 - lower_bound
    lower_probs = {}
    upper_probs = {}

    if not samples: 
        return lower_probs, upper_probs

    for from_regime, regime_samples in samples.items():
        if not isinstance(regime_samples, np.ndarray) or regime_samples.size == 0:
            num_to_regimes = regime_samples.shape[1] if regime_samples.ndim == 2 else 0
            lower_probs[from_regime] = np.full(num_to_regimes, np.nan)
            upper_probs[from_regime] = np.full(num_to_regimes, np.nan)
            continue
        
        if regime_samples.ndim == 1:
            pass

        lower_probs[from_regime] = np.quantile(regime_samples, lower_bound, axis=0)
        upper_probs[from_regime] = np.quantile(regime_samples, upper_bound, axis=0)
    return lower_probs, upper_probs

def identify_volatility_regimes_gmm(volatility, num_regimes_target=3, random_state=42):
    clean_vol = volatility.dropna()
    min_samples_needed = num_regimes_target * 5 
    
    default_regime_info = {
        'model': None, 'mapping': {}, 
        'means': [np.nan] * num_regimes_target, 
        'variances': [np.nan] * num_regimes_target, 
        'weights': [np.nan] * num_regimes_target
    }

    if len(clean_vol) < min_samples_needed or len(clean_vol.unique()) < num_regimes_target :
        regimes = pd.Series(np.nan, index=volatility.index)
        if not clean_vol.empty and num_regimes_target >= 1:
            regimes.loc[clean_vol.index] = 0 
            mean_val = clean_vol.mean()
            var_val = clean_vol.var(ddof=0) if len(clean_vol) > 0 else 0.0 
            
            current_regime_info = default_regime_info.copy()
            current_regime_info['mapping'] = {0:0}
            if num_regimes_target > 0:
                current_regime_info['means'][0] = mean_val
                current_regime_info['variances'][0] = var_val if not pd.isna(var_val) else np.nan
                current_regime_info['weights'][0] = 1.0
                for i in range(1, num_regimes_target):
                    current_regime_info['weights'][i] = 0.0
            return regimes, current_regime_info
        else: 
            return regimes, default_regime_info


    X = clean_vol.values.reshape(-1, 1)
    if np.all(X == X[0]): 
        regimes = pd.Series(np.nan, index=volatility.index)
        regimes.loc[clean_vol.index] = 0 
        current_regime_info = default_regime_info.copy()
        current_regime_info['mapping'] = {0:0}
        if num_regimes_target > 0:
            current_regime_info['means'][0] = X[0][0]
            current_regime_info['variances'][0] = 0.0
            current_regime_info['weights'][0] = 1.0
            for i in range(1, num_regimes_target):
                current_regime_info['weights'][i] = 0.0
        return regimes, current_regime_info

    gmm = GaussianMixture(
        n_components=num_regimes_target, covariance_type='full',
        random_state=random_state, n_init=10, reg_covar=1e-6 
    )
    try:
        gmm.fit(X)
        if not gmm.converged_:
            print(f"Warning: GMM did not converge for a stock ({clean_vol.name if hasattr(clean_vol, 'name') else 'Unnamed Volatility Series'}). Results might be suboptimal.")
    except ValueError as e: 
        print(f"GMM fitting error ({clean_vol.name if hasattr(clean_vol, 'name') else 'Unnamed Volatility Series'}): {e}. Falling back to single regime like behavior.")
        regimes = pd.Series(np.nan, index=volatility.index)
        regimes.loc[clean_vol.index] = 0
        mean_val = clean_vol.mean()
        var_val = clean_vol.var(ddof=0) if len(clean_vol) > 0 else 0.0
        current_regime_info = default_regime_info.copy()
        current_regime_info['mapping'] = {0:0}
        if num_regimes_target > 0:
            current_regime_info['means'][0] = mean_val
            current_regime_info['variances'][0] = var_val if not pd.isna(var_val) else np.nan
            current_regime_info['weights'][0] = 1.0
            for i in range(1, num_regimes_target):
                current_regime_info['weights'][i] = 0.0
        return regimes, current_regime_info

    regime_labels_raw = gmm.predict(X)
    
    gmm_means_flat = gmm.means_.flatten()
    regime_order = np.argsort(gmm_means_flat) 
    
    regime_mapping = {old_label: new_label for new_label, old_label in enumerate(regime_order)}
    regimes_clean = pd.Series(regime_labels_raw, index=clean_vol.index).map(regime_mapping) 
    
    regimes = pd.Series(np.nan, index=volatility.index) 
    regimes.loc[regimes_clean.index] = regimes_clean 
    
    actual_fitted_components = gmm.n_components 
    
    final_regime_info = {
        'model': gmm, 
        'mapping': regime_mapping, 
        'means': [gmm.means_[regime_order[i]][0] if i < actual_fitted_components else np.nan for i in range(num_regimes_target)],
        'variances': [gmm.covariances_[regime_order[i]][0][0] if i < actual_fitted_components else np.nan for i in range(num_regimes_target)],
        'weights': [gmm.weights_[regime_order[i]] if i < actual_fitted_components else 0.0 for i in range(num_regimes_target)]
    }
    return regimes, final_regime_info

def analyze_stock_with_gmm(symbol, data, window=20, num_regimes_target=3, prior_strength=1.0,
                           start_date=None, end_date=None): # Defaults changed to None
    if data is None or data.empty:
        return None

    # Apply date filtering only if start_date or end_date are specified by the caller
    if start_date is not None or end_date is not None:
        # If only one is None, use data's boundary for the other to avoid error in filter_data_by_date
        _start_date_str = start_date if start_date else data.index.min().strftime('%Y-%m-%d')
        _end_date_str = end_date if end_date else data.index.max().strftime('%Y-%m-%d')
        data_to_analyze = filter_data_by_date(data, _start_date_str, _end_date_str)
    else:
        data_to_analyze = data.copy() # Use data as is if no dates provided for filtering

    if data_to_analyze.empty:
        # Construct a minimal empty result if filtering leads to empty data
        base_index_for_empty = data.index if data is not None else pd.Index([])
        ohlc_cols_present_empty = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns] if data is not None else []
        ohlc_df_for_empty_results = data[ohlc_cols_present_empty].copy() if data is not None and ohlc_cols_present_empty else pd.DataFrame(index=base_index_for_empty)
        return {
            'symbol': symbol, 
            'ohlc_data_filtered': ohlc_df_for_empty_results, # original data's ohlc if data_to_analyze is empty
            'volatility': pd.Series(dtype=float, index=base_index_for_empty), 
            'regimes': pd.Series(dtype=float, index=base_index_for_empty), 
            'regime_info': {'means': [np.nan]*num_regimes_target, 
                            'variances': [np.nan]*num_regimes_target, 
                            'weights': [np.nan]*num_regimes_target}, 
            'transition_counts': np.zeros((num_regimes_target, num_regimes_target)),
            'transition_probabilities': np.full((num_regimes_target, num_regimes_target), np.nan),
            'lower_probabilities': {i: np.full(num_regimes_target, np.nan) for i in range(num_regimes_target)}, 
            'upper_probabilities': {i: np.full(num_regimes_target, np.nan) for i in range(num_regimes_target)},
            'current_regime': None, 
            'actual_num_regimes': 0,
            'regime_return_statistics': []
        }

    base_index = data_to_analyze.index 
    ohlc_cols_present = [col for col in ['open', 'high', 'low', 'close'] if col in data_to_analyze.columns]
    ohlc_df_for_results = data_to_analyze[ohlc_cols_present].copy() if ohlc_cols_present else pd.DataFrame(index=base_index)

    empty_results = {
        'symbol': symbol, 
        'ohlc_data_filtered': ohlc_df_for_results,
        'volatility': pd.Series(dtype=float, index=base_index), 
        'regimes': pd.Series(dtype=float, index=base_index), 
        'regime_info': {'means': [np.nan]*num_regimes_target, 
                        'variances': [np.nan]*num_regimes_target, 
                        'weights': [np.nan]*num_regimes_target}, 
        'transition_counts': np.zeros((num_regimes_target, num_regimes_target)),
        'transition_probabilities': np.full((num_regimes_target, num_regimes_target), np.nan),
        'lower_probabilities': {i: np.full(num_regimes_target, np.nan) for i in range(num_regimes_target)}, 
        'upper_probabilities': {i: np.full(num_regimes_target, np.nan) for i in range(num_regimes_target)},
        'current_regime': None, 
        'actual_num_regimes': 0,
        'regime_return_statistics': []
    }

    volatility = calculate_volatility(data_to_analyze, window) 
    if volatility is None or volatility.dropna().empty: 
        if volatility is not None: empty_results['volatility'] = volatility
        return empty_results 
    empty_results['volatility'] = volatility

    regimes, regime_info = identify_volatility_regimes_gmm(volatility, num_regimes_target)
    empty_results['regimes'] = regimes 
    empty_results['regime_info'] = regime_info 

    actual_num_regimes_found = 0
    if regime_info and regime_info.get('means'):
        actual_num_regimes_found = sum(1 for m in regime_info['means'][:num_regimes_target] if not pd.isna(m))
    empty_results['actual_num_regimes'] = actual_num_regimes_found

    if actual_num_regimes_found == 0:
         return empty_results 

    effective_dim_for_matrix = max(1, actual_num_regimes_found)

    tc_raw = count_transitions(regimes)
    num_distinct_labels_in_regimes = 0
    if not regimes.dropna().empty:
        max_label = regimes.dropna().max()
        if not pd.isna(max_label) and max_label >=0:
            num_distinct_labels_in_regimes = int(max_label) + 1
    
    dim_bayes = actual_num_regimes_found

    tc_for_bayes = np.zeros((dim_bayes, dim_bayes))
    if tc_raw.shape[0] > 0 and tc_raw.shape[1] > 0 : 
        common_dim = min(tc_raw.shape[0], dim_bayes)
        tc_for_bayes[:common_dim, :common_dim] = tc_raw[:common_dim, :common_dim]
    empty_results['transition_counts'] = tc_for_bayes
    
    mean_probs, posterior_samples = bayesian_transition_analysis(tc_for_bayes, prior_strength)
    if mean_probs.shape[0] != dim_bayes: 
        mean_probs = np.full((dim_bayes, dim_bayes), np.nan)
        posterior_samples = {i: np.array([]) for i in range(dim_bayes)}
    empty_results['transition_probabilities'] = mean_probs

    lower_probs, upper_probs = calculate_probability_intervals(posterior_samples)
    empty_results['lower_probabilities'] = {i: (lower_probs.get(i, np.full(dim_bayes, np.nan))) for i in range(dim_bayes)}
    empty_results['upper_probabilities'] = {i: (upper_probs.get(i, np.full(dim_bayes, np.nan))) for i in range(dim_bayes)}
    
    current_regime = None
    cleaned_regimes = regimes.dropna()
    if not cleaned_regimes.empty:
        last_regime_val = cleaned_regimes.iloc[-1]
        if not pd.isna(last_regime_val):
            current_regime = int(last_regime_val)
            if not (0 <= current_regime < actual_num_regimes_found):
                current_regime = None 
    empty_results['current_regime'] = current_regime
    
    if 'close' in data_to_analyze.columns and actual_num_regimes_found > 0:
        prices_for_returns = data_to_analyze['close']
        daily_log_ret = pd.Series(np.nan, index=prices_for_returns.index, dtype=float)
        shifted_prices = prices_for_returns.shift(1)
        valid_idx_ret = (prices_for_returns > 0) & (shifted_prices > 0) & (~prices_for_returns.isnull()) & (~shifted_prices.isnull())
        if valid_idx_ret.any():
            daily_log_ret[valid_idx_ret] = np.log(prices_for_returns[valid_idx_ret] / shifted_prices[valid_idx_ret])

        stats_data = pd.DataFrame({'LogReturn': daily_log_ret, 'Regime': regimes}).dropna()

        regime_stats_list = []
        if not stats_data.empty:
            for i in range(actual_num_regimes_found):
                regime_specific_returns = stats_data[stats_data['Regime'] == i]['LogReturn']
                n_days = len(regime_specific_returns)
                if n_days > 0:
                    mean_ret = regime_specific_returns.mean()
                    std_ret = regime_specific_returns.std()
                    skew_ret = regime_specific_returns.skew() if n_days > 2 else np.nan
                    kurt_ret = regime_specific_returns.kurtosis() if n_days > 3 else np.nan
                    total_ret_comp = np.exp(regime_specific_returns.sum()) - 1
                    
                    sharpe = np.nan
                    if std_ret > 0 and not pd.isna(std_ret) and not pd.isna(mean_ret):
                        # Ensure mean_ret is not excessively small leading to huge Sharpe due to precision
                        if abs(mean_ret) > 1e-9 : # Avoid division by tiny std or large mean*sqrt(252)
                             sharpe = (mean_ret * np.sqrt(252)) / std_ret if std_ret > 1e-9 else np.nan # Annualized Sharpe
                        else:
                             sharpe = 0.0 # If mean return is effectively zero
                    
                    regime_stats_list.append({
                        'Regime': i,
                        'N_Days': n_days,
                        'Mean_Daily_Log_Return': mean_ret,
                        'StdDev_Daily_Log_Return': std_ret,
                        'Annualized_Sharpe_Ratio': sharpe,
                        'Skewness_Daily_Log_Return': skew_ret,
                        'Kurtosis_Daily_Log_Return': kurt_ret,
                        'Total_Compounded_Log_Return': total_ret_comp
                    })
                else: 
                     regime_stats_list.append({
                        'Regime': i, 'N_Days': 0, 'Mean_Daily_Log_Return': np.nan,
                        'StdDev_Daily_Log_Return': np.nan, 'Annualized_Sharpe_Ratio': np.nan,
                        'Skewness_Daily_Log_Return': np.nan, 'Kurtosis_Daily_Log_Return': np.nan,
                        'Total_Compounded_Log_Return': np.nan
                    })
        empty_results['regime_return_statistics'] = regime_stats_list
    return empty_results

def save_stock_excel_report(results, symbol, save_dir):
    if results is None:
        # print(f"No results to save for {symbol}.")
        return

    output_path = os.path.join(save_dir, f"{symbol}_analysis_results.xlsx")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_vol_reg_ts = pd.DataFrame(index=results.get('ohlc_data_filtered', pd.DataFrame()).index)
            if df_vol_reg_ts.empty and results.get('volatility') is not None and not results.get('volatility').empty:
                 df_vol_reg_ts = pd.DataFrame(index=results.get('volatility').index)

            ohlc_data = results.get('ohlc_data_filtered')
            if ohlc_data is not None and not ohlc_data.empty:
                for col in ['open', 'high', 'low', 'close']: 
                    if col in ohlc_data.columns:
                        df_vol_reg_ts[col.capitalize()] = ohlc_data[col]
            
            volatility_series = results.get('volatility')
            if volatility_series is not None and not volatility_series.empty:
                df_vol_reg_ts['Volatility'] = volatility_series
            
            regimes_series = results.get('regimes')
            if regimes_series is not None and not regimes_series.empty:
                df_vol_reg_ts['Identified_Regime'] = regimes_series

            if not df_vol_reg_ts.empty:
                if isinstance(df_vol_reg_ts.index, pd.DatetimeIndex):
                     df_vol_reg_ts.index.name = 'Date'
                df_vol_reg_ts.to_excel(writer, sheet_name='Volatility_Regime_TS', index=True)
            else:
                pd.DataFrame([{"Status": "Time series data (OHLC, Vol, Regime) not available."}]).to_excel(writer, sheet_name='Volatility_Regime_TS', index=False)

            regime_info = results.get('regime_info', {})
            num_actual_regimes = results.get('actual_num_regimes', 0)
            regime_data_params = []
            if num_actual_regimes > 0 and regime_info.get('means'):
                for i in range(num_actual_regimes): 
                    mean_vol = regime_info['means'][i] if i < len(regime_info['means']) else np.nan
                    variance = regime_info['variances'][i] if i < len(regime_info['variances']) else np.nan
                    std_dev_vol = np.sqrt(variance) if not pd.isna(variance) and variance >=0 else np.nan
                    weight = regime_info['weights'][i] if i < len(regime_info['weights']) else np.nan
                    regime_data_params.append({
                        'Regime': i, 'Mean_Volatility': mean_vol,
                        'StdDev_Volatility': std_dev_vol, 'GMM_Weight': weight
                    })
            df_regime_params = pd.DataFrame(regime_data_params)
            if not df_regime_params.empty:
                df_regime_params.to_excel(writer, sheet_name='Regime_Parameters', index=False)
            else:
                pd.DataFrame([{"Status": "No GMM regime parameters identified."}]).to_excel(writer, sheet_name='Regime_Parameters', index=False)

            current_regime = results.get('current_regime')
            transition_probs_matrix = results.get('transition_probabilities')
            lower_probs_dict = results.get('lower_probabilities', {})
            upper_probs_dict = results.get('upper_probabilities', {})
            transition_data = []
            if current_regime is not None and \
               transition_probs_matrix is not None and transition_probs_matrix.ndim == 2 and \
               0 <= current_regime < transition_probs_matrix.shape[0] and \
               num_actual_regimes > 0 and transition_probs_matrix.shape[1] == num_actual_regimes:
                
                current_probs_row = transition_probs_matrix[current_regime]
                current_lower_row = lower_probs_dict.get(current_regime, np.full(num_actual_regimes, np.nan))
                current_upper_row = upper_probs_dict.get(current_regime, np.full(num_actual_regimes, np.nan))
                if not isinstance(current_lower_row, np.ndarray) or len(current_lower_row) != num_actual_regimes:
                    current_lower_row = np.full(num_actual_regimes, np.nan)
                if not isinstance(current_upper_row, np.ndarray) or len(current_upper_row) != num_actual_regimes:
                    current_upper_row = np.full(num_actual_regimes, np.nan)

                for to_regime in range(num_actual_regimes):
                    transition_data.append({
                        'From_Regime': current_regime, 'To_Regime': to_regime,
                        'Mean_Probability': current_probs_row[to_regime] if to_regime < len(current_probs_row) else np.nan,
                        '95%_CI_Lower': current_lower_row[to_regime] if to_regime < len(current_lower_row) else np.nan,
                        '95%_CI_Upper': current_upper_row[to_regime] if to_regime < len(current_upper_row) else np.nan,
                    })
            df_transitions = pd.DataFrame(transition_data)
            if not df_transitions.empty:
                df_header = pd.DataFrame([{"Current_Regime_Analyzed": current_regime if current_regime is not None else "N/A"}])
                df_header.to_excel(writer, sheet_name='Current_Regime_Transitions', index=False, header=True, startrow=0)
                df_transitions.to_excel(writer, sheet_name='Current_Regime_Transitions', index=False, startrow=2)
            else:
                pd.DataFrame([{"Status": "Transition data from current regime not available."}]).to_excel(writer, sheet_name='Current_Regime_Transitions', index=False)

            regime_return_stats_list = results.get('regime_return_statistics', [])
            if regime_return_stats_list: 
                df_regime_return_stats = pd.DataFrame(regime_return_stats_list)
                if not df_regime_return_stats.empty:
                    df_regime_return_stats.to_excel(writer, sheet_name='Regime_Return_Stats', index=False)
                else:
                    pd.DataFrame([{"Status": "Regime return statistics could not be computed or regimes had no data."}]).to_excel(writer, sheet_name='Regime_Return_Stats', index=False)
            else:
                 pd.DataFrame([{"Status": "No regime return statistics available."}]).to_excel(writer, sheet_name='Regime_Return_Stats', index=False)

    except Exception as e:
        print(f"Error saving Excel report for {symbol} to {output_path}: {e}")


# Worker function for multiprocessing calculations (used by run_gmm_analysis)
def process_stock_wrapper(args_tuple):
    symbol, data_df, window, num_regimes_target, prior_strength, start_date_str_arg, end_date_str_arg, intermediate_save_dir = args_tuple
    sd_str = start_date_str_arg.replace('-', '') if start_date_str_arg else "None"
    ed_str = end_date_str_arg.replace('-', '') if end_date_str_arg else "None"
    params_str = f"w{window}_r{num_regimes_target}_p{prior_strength}_sd{sd_str}_ed{ed_str}_v2"
    checkpoint_file = os.path.join(intermediate_save_dir, f"{symbol}_results_{params_str}.pkl")

    if os.path.exists(checkpoint_file): 
        try:
            with open(checkpoint_file, 'rb') as f:
                results = pickle.load(f)
            if results and isinstance(results, dict) and results.get('symbol') == symbol:
                if 'regime_return_statistics' in results and 'ohlc_data_filtered' in results:
                    return symbol, results
                else:
                    pass 
            else:
                pass 
        except Exception as e:
            print(f"Error loading cached results for {symbol} (params: {params_str}): {e}. Recomputing...")

    if data_df is None or data_df.empty : 
        return symbol, None 

    # Call analyze_stock_with_gmm with the explicit start/end dates from run_gmm_analysis
    results = analyze_stock_with_gmm(symbol, data_df, window, num_regimes_target, prior_strength,
                                     start_date_str_arg, end_date_str_arg)
    if results: 
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error saving intermediate results for {symbol} to {checkpoint_file}: {e}")
    return symbol, results

# Worker function for multiprocessing Excel report generation (used by run_gmm_analysis)
def save_stock_excel_report_wrapper(args_tuple):
    results_obj, symbol_key, report_save_dir = args_tuple
    if results_obj:
        save_stock_excel_report(results_obj, symbol_key, report_save_dir)
    return symbol_key 


def run_gmm_analysis(folder_path=DATA_PATH, window=20, num_regimes_target=3, prior_strength=1.0,
                   save_dir="volatility_excel_reports", start_date='2010-01-01', end_date=None, # Changed to None for end_date default
                   num_processes=None, force_recompute=False):
    print(f"Loading stock data from {folder_path}...")
    all_stock_data_loaded = load_stock_data(folder_path)
    if not all_stock_data_loaded:
        print("No valid stock data found after loading attempts.")
        return {}, []

    print(f"Found {len(all_stock_data_loaded)} stock datasets for potential analysis.")
    effective_end_date_str = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
    # Ensure start_date is also a string for consistency in process_stock_wrapper
    effective_start_date_str = start_date if start_date else '1900-01-01' # A very early default if None
    
    print(f"Analysis period: {effective_start_date_str} to {effective_end_date_str}")

    results_dir = save_dir
    os.makedirs(results_dir, exist_ok=True)
    intermediate_results_dir = os.path.join(results_dir, "intermediate_calc_results_v2") 
    os.makedirs(intermediate_results_dir, exist_ok=True)

    if force_recompute:
        print("Force recompute is ON. Deleting existing intermediate pkl results...")
        deleted_count = 0
        if os.path.exists(intermediate_results_dir):
            for f_name in os.listdir(intermediate_results_dir):
                if f_name.endswith(".pkl"): 
                    try:
                        os.remove(os.path.join(intermediate_results_dir, f_name))
                        deleted_count +=1
                    except Exception as e:
                        print(f"Error deleting {f_name}: {e}")
        print(f"Deleted {deleted_count} .pkl files from intermediate cache.")
    
    print("\n--- Starting Calculation Pass ---")
    tasks_calc = []
    for symbol, data_df_orig in all_stock_data_loaded.items():
        if data_df_orig is None or data_df_orig.empty: 
            continue
        tasks_calc.append((symbol, data_df_orig.copy(), window, num_regimes_target, prior_strength,
                      effective_start_date_str, effective_end_date_str, 
                      intermediate_results_dir))

    all_calculated_results = {}
    if not num_processes: 
        num_processes = max(1, os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1)
    
    print(f"Using {num_processes} processes for calculations.")

    if tasks_calc:
        if num_processes > 1 and len(tasks_calc) > 1: 
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool_results_calc = pool.map(process_stock_wrapper, tasks_calc)
        else: 
            print("Running calculations in sequential mode.")
            pool_results_calc = [process_stock_wrapper(task) for task in tasks_calc]
        
        for symbol_key, res_obj in pool_results_calc:
            if res_obj: 
                all_calculated_results[symbol_key] = res_obj
    
    print(f"\n--- Calculation Pass Complete. Got results for {len(all_calculated_results)} out of {len(tasks_calc)} potential stocks. ---")
    if not all_calculated_results:
        print("No stocks yielded results from the calculation pass. Exiting report generation.")
        return {}, []

    print("\n--- Starting Excel Report Generation Pass ---")
    tasks_report = []
    sorted_symbols_for_report = sorted(all_calculated_results.keys())
    for symbol in sorted_symbols_for_report:
        results = all_calculated_results[symbol]
        if results: 
            tasks_report.append((results, symbol, results_dir))

    if tasks_report:
        print(f"Using {num_processes} processes for Excel report generation.")
        if num_processes > 1 and len(tasks_report) > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                report_gen_symbols = pool.map(save_stock_excel_report_wrapper, tasks_report)
        else:
            print("Running Excel report generation in sequential mode.")
            for task in tasks_report:
                save_stock_excel_report_wrapper(task)
    print(f"\n--- Excel Report Generation Pass Complete for {len(tasks_report)} stocks. ---")


    print("\n--- Starting Overall Summary Generation ---")
    summary_data_list = []
    for symbol in sorted_symbols_for_report:
        results = all_calculated_results.get(symbol) 
        if results is None:
            continue
        
        current_regime = results.get('current_regime') 
        actual_num_regimes = results.get('actual_num_regimes', 0)
        regime_info = results.get('regime_info', {}) 
        transition_probs_matrix = results.get('transition_probabilities') 

        summary_entry = {'Symbol': symbol, 
                         'Current_Regime': current_regime if current_regime is not None else 'N/A',
                         'Actual_Num_Regimes': actual_num_regimes
                        }
        
        vol_series = results.get('volatility')
        if vol_series is not None and not vol_series.dropna().empty:
            summary_entry['Current_Raw_Volatility'] = vol_series.dropna().iloc[-1]
        else:
            summary_entry['Current_Raw_Volatility'] = np.nan

        if current_regime is not None and regime_info.get('means') and \
           actual_num_regimes > 0 and 0 <= current_regime < len(regime_info['means']) and \
           0 <= current_regime < actual_num_regimes : # Check current_regime is valid for the actual regimes
            summary_entry['Regime_Mean_Volatility_Current'] = regime_info['means'][current_regime]
        else:
            summary_entry['Regime_Mean_Volatility_Current'] = np.nan

        if current_regime is not None and transition_probs_matrix is not None and \
           transition_probs_matrix.ndim == 2 and 0 <= current_regime < transition_probs_matrix.shape[0] and \
           actual_num_regimes > 0 and transition_probs_matrix.shape[1] == actual_num_regimes:
            
            current_probs_row = transition_probs_matrix[current_regime]
            if 0 <= current_regime < actual_num_regimes: 
                 summary_entry['Prob_Stay_Same_Regime'] = current_probs_row[current_regime]
            else:
                summary_entry['Prob_Stay_Same_Regime'] = np.nan
            
            for i in range(actual_num_regimes): 
                summary_entry[f'Prob_To_Regime_{i}'] = current_probs_row[i] if i < len(current_probs_row) else np.nan
        else: 
            summary_entry['Prob_Stay_Same_Regime'] = np.nan
            # Pad with NaNs up to num_regimes_target for consistent column structure
            for i in range(num_regimes_target): 
                 summary_entry[f'Prob_To_Regime_{i}'] = np.nan
        
        if actual_num_regimes < num_regimes_target:
            for i in range(actual_num_regimes, num_regimes_target):
                summary_entry[f'Prob_To_Regime_{i}'] = np.nan


        for i in range(num_regimes_target): 
            if i < actual_num_regimes and regime_info.get('means') and i < len(regime_info['means']): 
                summary_entry[f'Regime_{i}_Mean_Vol'] = regime_info['means'][i]
                var = regime_info['variances'][i] if regime_info.get('variances') and i < len(regime_info['variances']) else np.nan
                summary_entry[f'Regime_{i}_StdDev_Vol'] = np.sqrt(var) if not pd.isna(var) and var >=0 else np.nan
                summary_entry[f'Regime_{i}_Weight'] = regime_info['weights'][i] if regime_info.get('weights') and i < len(regime_info['weights']) else np.nan
            else: 
                summary_entry[f'Regime_{i}_Mean_Vol'] = np.nan
                summary_entry[f'Regime_{i}_StdDev_Vol'] = np.nan
                summary_entry[f'Regime_{i}_Weight'] = np.nan
        
        regime_return_stats = results.get('regime_return_statistics', [])
        for i in range(num_regimes_target):
            stat_for_regime_i = next((stat for stat in regime_return_stats if stat['Regime'] == i), None)
            if stat_for_regime_i:
                summary_entry[f'Regime_{i}_Mean_Daily_LogRet'] = stat_for_regime_i.get('Mean_Daily_Log_Return', np.nan)
                summary_entry[f'Regime_{i}_Sharpe'] = stat_for_regime_i.get('Annualized_Sharpe_Ratio', np.nan)
                summary_entry[f'Regime_{i}_N_Days'] = stat_for_regime_i.get('N_Days', np.nan)
            else:
                summary_entry[f'Regime_{i}_Mean_Daily_LogRet'] = np.nan
                summary_entry[f'Regime_{i}_Sharpe'] = np.nan
                summary_entry[f'Regime_{i}_N_Days'] = np.nan

        summary_data_list.append(summary_entry)

    if summary_data_list:
        summary_df = pd.DataFrame(summary_data_list)
        if not summary_df.empty:
            cols_order = ['Symbol', 'Current_Regime', 'Actual_Num_Regimes', 'Current_Raw_Volatility', 'Regime_Mean_Volatility_Current', 'Prob_Stay_Same_Regime']
            for i in range(num_regimes_target): cols_order.append(f'Prob_To_Regime_{i}')
            for i in range(num_regimes_target):
                cols_order.append(f'Regime_{i}_Mean_Vol')
                cols_order.append(f'Regime_{i}_StdDev_Vol')
                cols_order.append(f'Regime_{i}_Weight')
                cols_order.append(f'Regime_{i}_N_Days')
                cols_order.append(f'Regime_{i}_Mean_Daily_LogRet')
                cols_order.append(f'Regime_{i}_Sharpe')
            
            final_cols = [col for col in cols_order if col in summary_df.columns]
            summary_df = summary_df[final_cols]
            summary_df.sort_values('Symbol', inplace=True)
            summary_path = os.path.join(results_dir, 'overall_analysis_summary_v2.xlsx') 
            try:
                summary_df.to_excel(summary_path, index=False, engine='openpyxl')
                print(f"\nOverall summary saved to {summary_path}")
            except Exception as e:
                print(f"Error saving overall summary Excel: {e}")
        else:
            print("Summary data list was populated but resulted in an empty DataFrame.")
    else:
        print("No summary data generated for the overall report.")

    print("\n--- Overall Summary Generation Complete ---")
    return all_calculated_results, summary_data_list


# --- START OF NEW BACKTESTING FUNCTIONS ---

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns series"""
    if cumulative_returns.empty:
        return np.nan
        
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max) - 1
    
    # Return the minimum (i.e., maximum drawdown)
    return drawdown.min()

def calculate_strategy_returns(test_data_with_regimes, train_results):
    """
    Calculate returns for a regime-based strategy vs buy and hold.
    
    Strategy: 
    - Invest 100% when in positive Sharpe regimes (from training)
    - Stay in cash when in negative Sharpe regimes
    
    Parameters:
    -----------
    test_data_with_regimes : pandas DataFrame
        Test data with LogReturn and Regime columns
    train_results : dict
        Results from analyze_stock_with_gmm on training data
        
    Returns:
    --------
    dict
        Strategy performance metrics
    """
    if test_data_with_regimes.empty:
        return {"strategy_return": np.nan, "buy_hold_return": np.nan, 
                "strategy_sharpe": np.nan, "buy_hold_sharpe": np.nan,
                "strategy_max_drawdown": np.nan, "buy_hold_max_drawdown": np.nan,
                "positive_sharpe_regimes": []}
    
    # Get regime statistics from training
    regime_stats = train_results.get('regime_return_statistics', [])
    
    # Identify positive Sharpe regimes
    positive_sharpe_regimes = set()
    for stat in regime_stats:
        # Ensure Sharpe is not NaN and is positive
        if stat.get('Annualized_Sharpe_Ratio') is not None and \
           not pd.isna(stat.get('Annualized_Sharpe_Ratio')) and \
           stat.get('Annualized_Sharpe_Ratio', -np.inf) > 0:
            positive_sharpe_regimes.add(stat.get('Regime'))
    
    # Create strategy returns
    strategy_returns = test_data_with_regimes['LogReturn'].copy() * 0  # Start with zeros (cash)
    
    # Only invest (use actual log returns) in positive Sharpe regimes
    for regime_idx in positive_sharpe_regimes:
        # Ensure regime_idx is not None before using it for masking
        if regime_idx is not None:
            regime_mask = test_data_with_regimes['Regime'] == regime_idx
            strategy_returns[regime_mask] = test_data_with_regimes.loc[regime_mask, 'LogReturn']
    
    # Calculate cumulative returns
    strategy_cumulative_log = strategy_returns.cumsum()
    buy_hold_cumulative_log = test_data_with_regimes['LogReturn'].cumsum()

    strategy_cumulative_simple = np.exp(strategy_cumulative_log) -1 # if log returns sum to 0, exp(0)-1 = 0
    buy_hold_cumulative_simple = np.exp(buy_hold_cumulative_log) -1


    # Calculate final returns
    strategy_return = strategy_cumulative_simple.iloc[-1] if not strategy_cumulative_simple.empty else np.nan
    buy_hold_return = buy_hold_cumulative_simple.iloc[-1] if not buy_hold_cumulative_simple.empty else np.nan
    
    # Calculate Sharpe ratios (annualized)
    # Ensure std_dev is not zero or very close to zero to avoid division errors or inf Sharpe
    strategy_std = strategy_returns.std()
    buy_hold_std = test_data_with_regimes['LogReturn'].std()

    strategy_sharpe = (strategy_returns.mean() * 252) / (strategy_std * np.sqrt(252)) \
        if strategy_std is not None and not pd.isna(strategy_std) and strategy_std > 1e-9 else np.nan
    buy_hold_sharpe = (test_data_with_regimes['LogReturn'].mean() * 252) / (buy_hold_std * np.sqrt(252)) \
        if buy_hold_std is not None and not pd.isna(buy_hold_std) and buy_hold_std > 1e-9 else np.nan
    
    # Maximum drawdown for strategy
    # calculate_max_drawdown expects simple cumulative returns (e.g., starting at 1)
    strategy_cum_returns_plot = np.exp(strategy_cumulative_log) # For drawdown calc
    buy_hold_cum_returns_plot = np.exp(buy_hold_cumulative_log) # For drawdown calc
    
    strategy_max_drawdown = calculate_max_drawdown(strategy_cum_returns_plot)
    buy_hold_max_drawdown = calculate_max_drawdown(buy_hold_cum_returns_plot)
    
    return {
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "strategy_sharpe": strategy_sharpe,
        "buy_hold_sharpe": buy_hold_sharpe,
        "strategy_max_drawdown": strategy_max_drawdown,
        "buy_hold_max_drawdown": buy_hold_max_drawdown,
        "positive_sharpe_regimes": list(positive_sharpe_regimes)
    }

def calculate_transition_accuracy(test_regimes, transition_probs):
    """
    Calculate how accurately the model predicted regime transitions.
    
    Parameters:
    -----------
    test_regimes : pandas Series
        Regimes identified on test data
    transition_probs : numpy.ndarray
        Transition probability matrix from training
        
    Returns:
    --------
    dict
        Transition prediction metrics
    """
    if not isinstance(transition_probs, np.ndarray) or transition_probs.ndim != 2 or transition_probs.shape[0] == 0:
        return {"accuracy": np.nan, "total_predictions": 0, "correct_predictions": 0, "confusion_matrix": np.array([])}
    
    # Clean test regimes
    regimes_clean = test_regimes.dropna().astype(int) # Ensure integer type for indexing
    if len(regimes_clean) < 2:
        return {"accuracy": np.nan, "total_predictions": 0, "correct_predictions": 0, "confusion_matrix": np.zeros_like(transition_probs)}
    
    correct_predictions = 0
    total_predictions = 0
    highest_prob_predictions_pairs = [] # Store (predicted, actual) pairs
    
    num_regimes_model = transition_probs.shape[0]

    # For each day, find the predicted next regime and actual next regime
    for i in range(len(regimes_clean) - 1):
        current_regime = regimes_clean.iloc[i]
        actual_next_regime = regimes_clean.iloc[i+1]
        
        # Skip if current regime is out of bounds for the trained model's transition_probs
        if not (0 <= current_regime < num_regimes_model):
            continue # This current_regime was not one the model was trained on (or an invalid label)
            
        # Get transition probabilities from current regime
        next_regime_probs_from_model = transition_probs[current_regime]
        
        # Predicted next regime is the one with highest probability from model
        predicted_next_regime = np.argmax(next_regime_probs_from_model)
        
        highest_prob_predictions_pairs.append((predicted_next_regime, actual_next_regime))
        if predicted_next_regime == actual_next_regime:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else np.nan
    
    # Construct confusion matrix based on the model's number of regimes
    confusion = np.zeros((num_regimes_model, num_regimes_model), dtype=int)
    
    for pred, actual in highest_prob_predictions_pairs:
        # Ensure predicted and actual are within the bounds of the confusion matrix
        if 0 <= pred < num_regimes_model and 0 <= actual < num_regimes_model:
            confusion[pred, actual] += 1 # Rows are predicted, Columns are actual
    
    return {
        "accuracy": accuracy,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "confusion_matrix": confusion
    }

def evaluate_model_performance(train_results, test_data, test_regimes, test_transitions, transition_probs):
    """
    Evaluate the performance of the volatility regime model on test data.
    (test_transitions from count_transitions on test_regimes is not directly used here,
     accuracy is based on transition_probs from training vs actual sequence in test_regimes)
    
    Parameters:
    -----------
    train_results : dict
        Results from analyze_stock_with_gmm on training data
    test_data : pandas DataFrame
        Test OHLC data
    test_regimes : pandas Series
        Regimes identified on test data
    test_transitions : numpy.ndarray 
        Actual transition counts in test data (more for observation, not direct use in this func)
    transition_probs : numpy.ndarray
        Transition probability matrix from training
        
    Returns:
    --------
    dict
        Performance metrics
    """
    # Calculate log returns for test data
    test_returns = pd.Series(np.nan, index=test_data.index)
    if 'close' in test_data.columns:
        prices = test_data['close']
        shifted_prices = prices.shift(1)
        valid_idx = (prices > 0) & (shifted_prices > 0) & (~prices.isnull()) & (~shifted_prices.isnull())
        if valid_idx.any():
            test_returns[valid_idx] = np.log(prices[valid_idx] / shifted_prices[valid_idx])
    
    # Combine returns with regimes
    test_data_with_regimes = pd.DataFrame({
        'LogReturn': test_returns,
        'Regime': test_regimes # test_regimes comes from GMM prediction on test_volatility
    }).dropna(subset=['LogReturn', 'Regime']) # Drop if EITHER is NaN for metric calculation
     # Ensure Regime is int for grouping
    if 'Regime' in test_data_with_regimes:
        test_data_with_regimes['Regime'] = test_data_with_regimes['Regime'].astype(int)

    
    # Calculate regime-specific metrics for test data
    regime_metrics = {}
    # num_regimes_from_train_model can be inferred from transition_probs or train_results
    num_regimes_in_model = 0
    if isinstance(transition_probs, np.ndarray) and transition_probs.ndim == 2:
        num_regimes_in_model = transition_probs.shape[0]
    elif train_results and 'regime_info' in train_results and 'means' in train_results['regime_info']:
         num_regimes_in_model = len([m for m in train_results['regime_info']['means'] if not pd.isna(m)])


    if num_regimes_in_model > 0 and not test_data_with_regimes.empty:
        for i in range(num_regimes_in_model): # Iterate up to the number of regimes defined by the trained model
            regime_data = test_data_with_regimes[test_data_with_regimes['Regime'] == i]
            
            metrics_for_this_regime = {
                'count': 0, 'mean_return': np.nan, 'std_return': np.nan,
                'annualized_sharpe': np.nan, 'total_return': np.nan
            }
            if len(regime_data) > 0:
                returns_in_regime = regime_data['LogReturn']
                metrics_for_this_regime['count'] = len(regime_data)
                metrics_for_this_regime['mean_return'] = returns_in_regime.mean()
                std_ret = returns_in_regime.std()
                metrics_for_this_regime['std_return'] = std_ret
                
                # Calculate Sharpe
                if std_ret is not None and not pd.isna(std_ret) and std_ret > 1e-9 and \
                   metrics_for_this_regime['mean_return'] is not None and not pd.isna(metrics_for_this_regime['mean_return']):
                    metrics_for_this_regime['annualized_sharpe'] = \
                        (metrics_for_this_regime['mean_return'] * 252) / (std_ret * np.sqrt(252))
                else: # Handle cases with zero or near-zero std dev or nan mean
                    if metrics_for_this_regime['mean_return'] is not None and not pd.isna(metrics_for_this_regime['mean_return']) and metrics_for_this_regime['mean_return'] == 0:
                         metrics_for_this_regime['annualized_sharpe'] = 0.0
                    else: # Could be large positive or negative if std is tiny, or NaN
                         metrics_for_this_regime['annualized_sharpe'] = np.nan


                metrics_for_this_regime['total_return'] = np.exp(returns_in_regime.sum()) - 1
            regime_metrics[i] = metrics_for_this_regime
    
    # Calculate transition prediction accuracy using the TRAINED transition_probs
    transition_accuracy_metrics = calculate_transition_accuracy(test_regimes, transition_probs)
    
    # Calculate cumulative returns using regime-based strategy vs. buy and hold
    # Pass test_data_with_regimes (which has LogReturn and predicted Regime for test period)
    # and train_results (which has the Sharpe ratios from the training period used for the strategy decision)
    strategy_return_metrics = calculate_strategy_returns(test_data_with_regimes, train_results)
    
    return {
        "regime_performance_on_test": regime_metrics, # Renamed for clarity
        "transition_accuracy": transition_accuracy_metrics,
        "strategy_returns": strategy_return_metrics
    }

def aggregate_backtest_results(validation_results):
    """Aggregate results across all validation periods"""
    if not validation_results:
        return {}
    
    # Extract key metrics
    accuracies = []
    strategy_returns_list = [] # Renamed to avoid conflict
    buy_hold_returns_list = [] # Renamed
    strategy_sharpes = []
    buy_hold_sharpes = []
    strategy_drawdowns = []
    buy_hold_drawdowns = []
    
    for result in validation_results:
        # Check if result is a dict and has the expected keys
        if not isinstance(result, dict): continue

        if 'transition_accuracy' in result and isinstance(result['transition_accuracy'], dict):
            accuracies.append(result['transition_accuracy'].get('accuracy', np.nan))
        
        if 'strategy_returns' in result and isinstance(result['strategy_returns'], dict):
            sr = result['strategy_returns']
            strategy_returns_list.append(sr.get('strategy_return', np.nan))
            buy_hold_returns_list.append(sr.get('buy_hold_return', np.nan))
            strategy_sharpes.append(sr.get('strategy_sharpe', np.nan))
            buy_hold_sharpes.append(sr.get('buy_hold_sharpe', np.nan))
            strategy_drawdowns.append(sr.get('strategy_max_drawdown', np.nan))
            buy_hold_drawdowns.append(sr.get('buy_hold_max_drawdown', np.nan))
    
    # Remove NaNs before aggregation
    accuracies = [x for x in accuracies if not pd.isna(x)]
    strategy_returns_list = [x for x in strategy_returns_list if not pd.isna(x)]
    buy_hold_returns_list = [x for x in buy_hold_returns_list if not pd.isna(x)]
    strategy_sharpes = [x for x in strategy_sharpes if not pd.isna(x)]
    buy_hold_sharpes = [x for x in buy_hold_sharpes if not pd.isna(x)]
    strategy_drawdowns = [x for x in strategy_drawdowns if not pd.isna(x)]
    buy_hold_drawdowns = [x for x in buy_hold_drawdowns if not pd.isna(x)]
    
    # Aggregate
    return {
        "mean_transition_accuracy": np.mean(accuracies) if accuracies else np.nan,
        "mean_strategy_return": np.mean(strategy_returns_list) if strategy_returns_list else np.nan,
        "mean_buy_hold_return": np.mean(buy_hold_returns_list) if buy_hold_returns_list else np.nan,
        "mean_strategy_sharpe": np.mean(strategy_sharpes) if strategy_sharpes else np.nan,
        "mean_buy_hold_sharpe": np.mean(buy_hold_sharpes) if buy_hold_sharpes else np.nan,
        "mean_strategy_drawdown": np.mean(strategy_drawdowns) if strategy_drawdowns else np.nan,
        "mean_buy_hold_drawdown": np.mean(buy_hold_drawdowns) if buy_hold_drawdowns else np.nan,
        "win_rate": sum(1 for s, b in zip(strategy_returns_list, buy_hold_returns_list) 
                         if s > b) / len(strategy_returns_list) if strategy_returns_list else np.nan,
        "number_of_validations_aggregated": len(accuracies) # Count based on one of the lists, e.g. accuracies
    }

def backtest_regime_model(stock_data, symbol, 
                          window=20, num_regimes=3, prior_strength=1.0,
                          train_ratio=0.6, validation_periods=5, 
                          min_train_days=252, random_seed=42):
    """
    Backtest the volatility regime model using out-of-sample validation periods.
    (train_ratio is part of signature but not used in random period selection)
    """
    np.random.seed(random_seed) # Set seed for reproducibility of random cutoffs
    
    if stock_data is None or stock_data.empty:
        return {"status": "error", "message": "Empty stock data"}
    
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        return {"status": "error", "message": "Stock data must have DatetimeIndex"}
    
    if len(stock_data) < min_train_days * 2:
        return {"status": "error", "message": f"Insufficient data: {len(stock_data)} days available, need at least {min_train_days * 2}"}
    
    validation_results = []
    
    for period_num in range(validation_periods):
        # Randomly select a cutoff point
        # Max cutoff leaves at least min_train_days/2 for testing (or a smaller fixed amount like window*2)
        # Min cutoff ensures train data has at least min_train_days
        # A test set needs at least `window` days for volatility calc, plus a bit more for transitions.
        min_test_days = max(window * 2, 30) # e.g. 2*volatility window or 30 days.
        
        max_cutoff_index = len(stock_data) - min_test_days 
        # min_cutoff_index ensures train_data has at least min_train_days
        # Also, train_data needs enough data for initial volatility calculation (e.g. window days)
        min_cutoff_index = min_train_days 

        if max_cutoff_index <= min_cutoff_index:
            # This can happen if total data is not much larger than min_train_days + min_test_days
            print(f"Warning for {symbol}, period {period_num}: Not enough range for random split (max_cutoff_idx {max_cutoff_index} <= min_cutoff_idx {min_cutoff_index}). Skipping period.")
            continue # Skip this validation period if range is invalid
        
        # cutoff is the length of the training set
        cutoff = np.random.randint(min_cutoff_index, max_cutoff_index + 1) # +1 because randint is exclusive for upper bound
        
        train_data = stock_data.iloc[:cutoff].copy()
        test_data = stock_data.iloc[cutoff:].copy()
        
        # Redundant check, but good safeguard
        if len(train_data) < min_train_days or len(test_data) < min_test_days:
            print(f"Warning for {symbol}, period {period_num}: Insufficient train/test data after split. Train: {len(train_data)}, Test: {len(test_data)}. Skipping.")
            continue
        
        # Train the model on training data
        # Calling analyze_stock_with_gmm without start_date/end_date, so it uses train_data as is
        train_results = analyze_stock_with_gmm(
            symbol=f"{symbol}_train_period_{period_num}", 
            data=train_data, # This is already sliced, analyze_stock_with_gmm will use it directly
            window=window,
            num_regimes_target=num_regimes,
            prior_strength=prior_strength
            # No start_date/end_date, so analyze_stock_with_gmm uses train_data as is
        )
        
        if train_results is None or 'regimes' not in train_results or train_results.get('actual_num_regimes', 0) == 0:
            print(f"Warning for {symbol}, period {period_num}: Training failed or yielded no regimes. Skipping.")
            continue
        
        gmm_model = train_results.get('regime_info', {}).get('model')
        if gmm_model is None: #This can happen if GMM failed or fell back to single regime without a formal model object
            print(f"Warning for {symbol}, period {period_num}: No GMM model object in training results. Skipping.")
            # If it fell back to single regime, actual_num_regimes might be 1, but model object might be None.
            # We need the model to predict on test data. If it's a fallback, predicting on test data might mean assigning all to regime 0.
            # For simplicity now, skip if no explicit GMM model.
            # A more robust fallback could be created if needed (e.g. if actual_num_regimes is 1, predict 0 for all test data).
            continue
            
        regime_mapping_from_train = train_results.get('regime_info', {}).get('mapping', {})
        # transition_probs from training are based on actual_num_regimes found in training
        trained_transition_probs = train_results.get('transition_probabilities', np.array([])) 
        
        # Calculate volatility for test data
        test_volatility = calculate_volatility(test_data, window=window)
        test_vol_clean = test_volatility.dropna()

        if test_vol_clean.empty:
            print(f"Warning for {symbol}, period {period_num}: Test volatility is all NaN. Skipping.")
            continue
            
        X_test = test_vol_clean.values.reshape(-1, 1)
        try:
            test_regime_labels_raw = gmm_model.predict(X_test)
        except Exception as e:
            print(f"Error predicting regimes on test data for {symbol}, period {period_num}: {e}. Skipping.")
            continue

        # Map raw labels to sorted (0=low vol, 1=mid vol, etc.) using training mapping
        test_regimes_clean = pd.Series(test_regime_labels_raw, index=test_vol_clean.index).map(regime_mapping_from_train)
        
        # Create a full series of test regimes, including NaNs where volatility was NaN
        test_regimes_final = pd.Series(np.nan, index=test_volatility.index)
        test_regimes_final.loc[test_regimes_clean.index] = test_regimes_clean
        
        # Calculate actual transitions observed in test data (for informational purposes if needed)
        # test_transitions_observed = count_transitions(test_regimes_final) 
        # Not strictly needed by evaluate_model_performance, but could be logged.
        
        # Calculate performance metrics
        period_eval_results = evaluate_model_performance(
            train_results=train_results, # Contains training regime stats (e.g., Sharpe for strategy)
            test_data=test_data,         # Raw OHLC test data
            test_regimes=test_regimes_final, # Predicted regimes on test data
            test_transitions=None, # test_transitions_observed - Not directly used by evaluate func
            transition_probs=trained_transition_probs # From training
        )
        
        period_eval_results.update({
            "period_num": period_num,
            "train_start_date": train_data.index[0].strftime('%Y-%m-%d'),
            "train_end_date": train_data.index[-1].strftime('%Y-%m-%d'),
            "test_start_date": test_data.index[0].strftime('%Y-%m-%d'),
            "test_end_date": test_data.index[-1].strftime('%Y-%m-%d'),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "num_regimes_in_train_model": train_results.get('actual_num_regimes', 0)
        })
        
        validation_results.append(period_eval_results)
    
    if not validation_results:
         return {"status": "error", "message": f"No validation periods completed successfully for {symbol}."}

    aggregated_results = aggregate_backtest_results(validation_results)
    # Add total number of successful validation periods to aggregated results
    aggregated_results["number_of_successful_validations"] = len(validation_results)


    return {
        "status": "success",
        "validation_periods_results": validation_results,
        "aggregated_results": aggregated_results
    }

def create_backtest_summary(all_backtest_results):
    """Create summary of backtest results across all stocks"""
    stock_summaries = []
    
    for symbol, result in all_backtest_results.items():
        if not isinstance(result, dict) or result.get('status') != 'success':
            stock_summaries.append({
                'Symbol': symbol,
                'Status': result.get('status', 'unknown_error'),
                'Message': result.get('message', 'N/A'),
                'Validation_Periods_Attempted': result.get('validation_periods_results', []) if isinstance(result.get('validation_periods_results'), list) else 0, # Placeholder
                'Successful_Validations': 0,
                'Transition_Accuracy': np.nan, 'Strategy_Return': np.nan, 'Buy_Hold_Return': np.nan,
                'Return_Difference': np.nan, 'Strategy_Sharpe': np.nan, 'Buy_Hold_Sharpe': np.nan,
                'Strategy_MaxDD': np.nan, 'Buy_Hold_MaxDD': np.nan, 'Win_Rate': np.nan
            })
            continue
            
        agg = result.get('aggregated_results', {})
        
        stock_summaries.append({
            'Symbol': symbol,
            'Status': 'success',
            'Message': 'OK',
            'Validation_Periods_Attempted': len(result.get('validation_periods_results', [])), # Total periods attempted or where results were stored
            'Successful_Validations': agg.get('number_of_successful_validations', agg.get('number_of_validations_aggregated',0) ), # Number of periods that made it to aggregation
            'Transition_Accuracy': agg.get('mean_transition_accuracy', np.nan),
            'Strategy_Return': agg.get('mean_strategy_return', np.nan),
            'Buy_Hold_Return': agg.get('mean_buy_hold_return', np.nan),
            'Return_Difference': (agg.get('mean_strategy_return', np.nan) - agg.get('mean_buy_hold_return', np.nan)) 
                                 if not pd.isna(agg.get('mean_strategy_return')) and not pd.isna(agg.get('mean_buy_hold_return')) else np.nan,
            'Strategy_Sharpe': agg.get('mean_strategy_sharpe', np.nan),
            'Buy_Hold_Sharpe': agg.get('mean_buy_hold_sharpe', np.nan),
            'Strategy_MaxDD': agg.get('mean_strategy_drawdown', np.nan),
            'Buy_Hold_MaxDD': agg.get('mean_buy_hold_drawdown', np.nan),
            'Win_Rate': agg.get('win_rate', np.nan)
        })
    
    if not stock_summaries:
        return {'stock_summaries': [], 'overall_metrics': {}}

    summary_df = pd.DataFrame(stock_summaries)
    
    # Overall metrics - calculated from successfully processed stocks
    successful_summary_df = summary_df[summary_df['Status'] == 'success'].copy()
    
    overall_metrics = {
        'total_stocks_input': len(all_backtest_results),
        'total_stocks_successfully_backtested': len(successful_summary_df),
        'avg_successful_validations_per_stock': successful_summary_df['Successful_Validations'].mean() if not successful_summary_df.empty else np.nan,
        'avg_transition_accuracy': successful_summary_df['Transition_Accuracy'].mean(skipna=True) if not successful_summary_df.empty else np.nan,
        'avg_strategy_return': successful_summary_df['Strategy_Return'].mean(skipna=True) if not successful_summary_df.empty else np.nan,
        'avg_buy_hold_return': successful_summary_df['Buy_Hold_Return'].mean(skipna=True) if not successful_summary_df.empty else np.nan,
        'avg_return_difference': successful_summary_df['Return_Difference'].mean(skipna=True) if not successful_summary_df.empty else np.nan,
        'avg_strategy_sharpe': successful_summary_df['Strategy_Sharpe'].mean(skipna=True) if not successful_summary_df.empty else np.nan,
        'avg_buy_hold_sharpe': successful_summary_df['Buy_Hold_Sharpe'].mean(skipna=True) if not successful_summary_df.empty else np.nan,
        'pct_stocks_strategy_beats_buyhold_return': (successful_summary_df['Return_Difference'] > 0).mean() * 100 if not successful_summary_df.empty and 'Return_Difference' in successful_summary_df else np.nan,
        'pct_stocks_strategy_beats_buyhold_sharpe': ((successful_summary_df['Strategy_Sharpe'] > successful_summary_df['Buy_Hold_Sharpe']) & (successful_summary_df['Strategy_Sharpe'].notna() & successful_summary_df['Buy_Hold_Sharpe'].notna())).mean() * 100 if not successful_summary_df.empty and 'Strategy_Sharpe' in successful_summary_df else np.nan,
        'median_win_rate': successful_summary_df['Win_Rate'].median(skipna=True) if not successful_summary_df.empty else np.nan,
    }
    
    return {
        'stock_summaries': stock_summaries, # Full list including errors
        'overall_metrics': overall_metrics
    }

def run_backtest_analysis(folder_path, 
                          window=20, 
                          num_regimes=3, 
                          prior_strength=1.0,
                          validation_periods=5,
                          min_train_days=252,
                          random_seed_base=42, # Base seed, will be incremented for each stock
                          save_dir="backtest_results"):
    """
    Run backtesting analysis on all stocks in the folder.
    """
    print(f"Loading stock data from {folder_path} for backtesting...")
    all_stock_data = load_stock_data(folder_path)
    
    if not all_stock_data:
        print("No valid stock data found for backtesting.")
        return {}
    
    print(f"Found {len(all_stock_data)} stocks for backtesting.")
    
    os.makedirs(save_dir, exist_ok=True)
    all_backtest_results = {}
    
    stock_counter = 0
    for symbol, data in all_stock_data.items():
        print(f"\nBacktesting {symbol} ({stock_counter + 1}/{len(all_stock_data)})...")
        
        # Use a different seed for each stock's set of validation periods for diversity,
        # but keep the overall process reproducible if random_seed_base is fixed.
        stock_specific_seed = random_seed_base + stock_counter

        backtest_result = backtest_regime_model(
            stock_data=data,
            symbol=symbol,
            window=window,
            num_regimes=num_regimes,
            prior_strength=prior_strength,
            validation_periods=validation_periods,
            min_train_days=min_train_days,
            random_seed=stock_specific_seed # Pass the specific seed
        )
        
        all_backtest_results[symbol] = backtest_result
        
        try:
            # Save detailed results for this stock (includes all validation period details)
            with open(os.path.join(save_dir, f"{symbol}_backtest_details.pkl"), 'wb') as f:
                pickle.dump(backtest_result, f)
        except Exception as e:
            print(f"Error saving detailed backtest result for {symbol}: {e}")
        stock_counter += 1
    
    print("\nCreating overall backtest summary...")
    overall_summary = create_backtest_summary(all_backtest_results)
    
    try:
        with open(os.path.join(save_dir, "overall_backtest_summary_data.pkl"), 'wb') as f:
            pickle.dump(overall_summary, f)
        
        if overall_summary.get('stock_summaries'):
            df_summary = pd.DataFrame(overall_summary['stock_summaries'])
            df_summary.to_excel(
                os.path.join(save_dir, "overall_backtest_summary_report.xlsx"),
                index=False
            )
            print(f"Overall backtest summary report saved to Excel.")
        else:
            print("No stock summaries to save to Excel.")

    except Exception as e:
        print(f"Error saving overall backtest summary: {e}")
    
    print(f"\nBacktesting complete! Results saved to {os.path.abspath(save_dir)}")
    return overall_summary

# --- END OF NEW BACKTESTING FUNCTIONS ---


if __name__ == "__main__":
    data_folder_main = DATA_PATH 
    
    # Parameters for the original GMM analysis (if you want to run it)
    # output_gmm_analysis_dir = "volatility_analysis_excel_output_v6"
    # window_gmm = 20
    # regimes_gmm = 3
    # prior_gmm = 1.0
    # start_gmm = '2010-01-01'
    # end_gmm = None 
    # processes_gmm = None 
    # recompute_gmm = False

    # print("--- Running Original GMM Full Period Analysis (Example Call) ---")
    # print("Note: This part is commented out by default. Uncomment to run.")
    # results_dict_gmm, summary_list_gmm = run_gmm_analysis(
    #     folder_path=data_folder_main,
    #     window=window_gmm,
    #     num_regimes_target=regimes_gmm,
    #     prior_strength=prior_gmm,
    #     save_dir=output_gmm_analysis_dir,
    #     start_date=start_gmm,
    #     end_date=end_gmm,
    #     num_processes=processes_gmm,
    #     force_recompute=recompute_gmm
    # )
    # if results_dict_gmm:
    #     print(f"GMM Analysis: Successfully processed {len(results_dict_gmm)} stocks.")
    # print("--- Original GMM Full Period Analysis Complete ---")


    print("\n\n--- Running Backtesting Analysis ---")
    # Parameters for Backtesting
    backtest_output_dir = "volatility_model_backtest_results_v1"
    backtest_window = 20
    backtest_num_regimes = 3
    backtest_prior_strength = 1.0
    backtest_validation_periods = 10 # Test more periods for robustness
    backtest_min_train_days = 252 * 2 # e.g., 2 years of training data
    backtest_random_seed = 123 # For reproducibility of the entire backtest run

    print(f"Backtesting Parameters:")
    print(f"- Data folder: {data_folder_main}")
    print(f"- Output directory: {backtest_output_dir}")
    print(f"- Volatility window: {backtest_window}")
    print(f"- Target GMM regimes: {backtest_num_regimes}")
    print(f"- Bayesian prior strength: {backtest_prior_strength}")
    print(f"- Number of random validation periods per stock: {backtest_validation_periods}")
    print(f"- Minimum training days per period: {backtest_min_train_days}")
    print(f"- Base random seed: {backtest_random_seed}")

    backtest_summary_results = run_backtest_analysis(
        folder_path=data_folder_main,
        window=backtest_window,
        num_regimes=backtest_num_regimes,
        prior_strength=backtest_prior_strength,
        validation_periods=backtest_validation_periods,
        min_train_days=backtest_min_train_days,
        random_seed_base=backtest_random_seed,
        save_dir=backtest_output_dir
    )

    if backtest_summary_results and 'overall_metrics' in backtest_summary_results:
        print("\nOverall Backtest Metrics Summary:")
        for metric, value in backtest_summary_results['overall_metrics'].items():
            if isinstance(value, float):
                print(f"- {metric.replace('_', ' ').capitalize()}: {value:.4f}")
            else:
                print(f"- {metric.replace('_', ' ').capitalize()}: {value}")
    else:
        print("Backtesting finished, but no overall summary metrics were generated.")
    
    print("--- Backtesting Analysis Complete ---")