import pandas as pd
import numpy as np
import glob
import os
import math
# import scipy.stats as stats # No longer directly used, np and pd methods cover it
from datetime import datetime
from sklearn.mixture import GaussianMixture
# from scipy.stats import norm # norm from scipy.stats is not explicitly used here.
import pickle # For saving/loading intermediate results
import multiprocessing # For parallel processing
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

def filter_data_by_date(data, start_date_str='2010-01-01', end_date_str=None): # Renamed parameters for clarity
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

        if np.all(posterior_alphas <= 0) or posterior_alphas.sum() == 0 : 
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
                           start_date=None, end_date=None, random_state_gmm=42): # Added random_state_gmm
    if data is None or data.empty:
        return None

    # Apply date filtering only if start_date or end_date are specified by the caller
    if start_date is not None or end_date is not None:
        _start_date_str = start_date if start_date else data.index.min().strftime('%Y-%m-%d')
        _end_date_str = end_date if end_date else data.index.max().strftime('%Y-%m-%d')
        data_to_analyze = filter_data_by_date(data, _start_date_str, _end_date_str)
    else:
        data_to_analyze = data.copy() 

    if data_to_analyze.empty:
        base_index_for_empty = data.index if data is not None else pd.Index([])
        ohlc_cols_present_empty = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns] if data is not None else []
        ohlc_df_for_empty_results = data[ohlc_cols_present_empty].copy() if data is not None and ohlc_cols_present_empty else pd.DataFrame(index=base_index_for_empty)
        return {
            'symbol': symbol, 
            'ohlc_data_filtered': ohlc_df_for_empty_results,
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

    regimes, regime_info = identify_volatility_regimes_gmm(volatility, num_regimes_target, random_state=random_state_gmm) # Pass random_state
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
                        if abs(mean_ret) > 1e-9 : 
                             sharpe = (mean_ret * np.sqrt(252)) / std_ret if std_ret > 1e-9 else np.nan 
                        else:
                             sharpe = 0.0 
                    
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


# Worker function for multiprocessing calculations
def process_stock_wrapper(args_tuple):
    symbol, data_df, window, num_regimes_target, prior_strength, start_date_str_arg, end_date_str_arg, intermediate_save_dir, random_state_gmm = args_tuple # Added random_state_gmm
    sd_str = start_date_str_arg.replace('-', '') if start_date_str_arg else "None"
    ed_str = end_date_str_arg.replace('-', '') if end_date_str_arg else "None"
    params_str = f"w{window}_r{num_regimes_target}_p{prior_strength}_sd{sd_str}_ed{ed_str}_rs{random_state_gmm}_v3" # Added random_state to params_str, incremented version
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

    results = analyze_stock_with_gmm(symbol, data_df, window, num_regimes_target, prior_strength,
                                     start_date_str_arg, end_date_str_arg, random_state_gmm=random_state_gmm) # Pass random_state_gmm
    if results: 
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error saving intermediate results for {symbol} to {checkpoint_file}: {e}")
    return symbol, results

# Worker function for multiprocessing Excel report generation
def save_stock_excel_report_wrapper(args_tuple):
    results_obj, symbol_key, report_save_dir = args_tuple
    if results_obj:
        save_stock_excel_report(results_obj, symbol_key, report_save_dir)
    return symbol_key 


def run_gmm_analysis(folder_path=DATA_PATH, window=20, num_regimes_target=3, prior_strength=1.0,
                   save_dir="volatility_excel_reports", start_date='2010-01-01', end_date=None,
                   num_processes=None, force_recompute=False, random_state_gmm_base=42): # Added random_state_gmm_base
    print(f"Loading stock data from {folder_path}...")
    all_stock_data_loaded = load_stock_data(folder_path)
    if not all_stock_data_loaded:
        print("No valid stock data found after loading attempts.")
        return {}, []

    print(f"Found {len(all_stock_data_loaded)} stock datasets for potential analysis.")
    effective_end_date_str = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
    effective_start_date_str = start_date if start_date else '1900-01-01' 
    
    print(f"Analysis period: {effective_start_date_str} to {effective_end_date_str}")

    results_dir = save_dir
    os.makedirs(results_dir, exist_ok=True)
    intermediate_results_dir = os.path.join(results_dir, "intermediate_calc_results_v3") # Updated version
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
    stock_counter = 0
    for symbol, data_df_orig in all_stock_data_loaded.items():
        if data_df_orig is None or data_df_orig.empty: 
            continue
        current_random_state_gmm = random_state_gmm_base + stock_counter # Vary random state per stock
        tasks_calc.append((symbol, data_df_orig.copy(), window, num_regimes_target, prior_strength,
                      effective_start_date_str, effective_end_date_str, 
                      intermediate_results_dir, current_random_state_gmm)) # Pass current_random_state_gmm
        stock_counter += 1

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
           0 <= current_regime < actual_num_regimes : 
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
            summary_path = os.path.join(results_dir, 'overall_analysis_summary_v3.xlsx') # Updated version
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


if __name__ == "__main__":
    data_folder_main = DATA_PATH 
    
    # --- Parameters for GMM Full Period Analysis ---
    output_gmm_analysis_dir = "volatility_analysis_excel_output_v3_robust" # New output folder
    window_gmm = 20
    regimes_gmm = 3
    prior_gmm = 1.0
    start_gmm = '2010-01-01'
    end_gmm = None # Analyze up to the latest data
    processes_gmm = None # Use default (CPU count - 1)
    recompute_gmm = False # Set to True to ignore cached intermediate files
    gmm_analysis_random_state_base = 42 # Base random state for GMM, varied per stock

    print("--- Running GMM Full Period Analysis ---")
    print(f"Analysis Parameters:")
    print(f"- Data folder: {data_folder_main}")
    print(f"- Output directory: {output_gmm_analysis_dir}")
    print(f"- Volatility window: {window_gmm} days")
    print(f"- Target number of GMM regimes: {regimes_gmm}")
    print(f"- Prior strength for Bayesian transitions: {prior_gmm}")
    print(f"- Analysis period: {start_gmm} to {end_gmm if end_gmm else 'latest in data'}")
    print(f"- Number of parallel processes: {'Default (CPU count - 1)' if processes_gmm is None else processes_gmm}")
    print(f"- Force recomputation of intermediate files: {recompute_gmm}")
    print(f"- GMM random_state base: {gmm_analysis_random_state_base}")


    results_dict_gmm, summary_list_gmm = run_gmm_analysis(
        folder_path=data_folder_main,
        window=window_gmm,
        num_regimes_target=regimes_gmm,
        prior_strength=prior_gmm,
        save_dir=output_gmm_analysis_dir,
        start_date=start_gmm,
        end_date=end_gmm,
        num_processes=processes_gmm,
        force_recompute=recompute_gmm,
        random_state_gmm_base=gmm_analysis_random_state_base
    )
    
    print("\nAnalysis complete!")
    if results_dict_gmm:
        print(f"Successfully processed and generated data for {len(results_dict_gmm)} stocks.")
    if summary_list_gmm: 
        print(f"Generated overall summary data for {len(summary_list_gmm)} stocks.")
    print(f"All Excel reports and summary saved in: {os.path.abspath(output_gmm_analysis_dir)}")
    print("--- GMM Full Period Analysis Complete ---")