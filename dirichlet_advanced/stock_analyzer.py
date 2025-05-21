# stock_analyzer.py
"""
Core analysis function for a single stock, now with selectable volatility models
and improved transition/Bayesian analysis.
"""
import pandas as pd
import numpy as np
from data_utils import filter_data_by_date
from volatility_calculator import calculate_volatility as calculate_simple_volatility
from volatility_models import (calculate_garch_volatility, calculate_sv_volatility,
                               calculate_ewma_volatility, calculate_nn_volatility,
                               calculate_hybrid_volatility)
from gmm_regime_identifier import identify_volatility_regimes_gmm # Using 1D GMM
from transition_analyzer import (count_transitions_improved,
                                 bayesian_transition_analysis_improved,
                                 calculate_transition_probability_distributions)
from financial_metrics import calculate_regime_return_statistics
import config # For defaults

def _get_empty_results_template(symbol: str, data_for_index: pd.DataFrame, num_regimes_target: int) -> dict:
    base_idx = data_for_index.index if data_for_index is not None and hasattr(data_for_index, 'index') else pd.Index([])
    ohlc_cols = ['open', 'high', 'low', 'close']
    ohlc_df = pd.DataFrame(index=base_idx)
    if data_for_index is not None:
        for col in ohlc_cols:
            ohlc_df[col] = data_for_index[col] if col in data_for_index.columns else np.nan
    else:
        for col in ohlc_cols: ohlc_df[col] = np.nan
    
    nrt = num_regimes_target 
    return {
        'symbol': symbol,
        'ohlc_data_filtered': ohlc_df,
        'volatility': pd.Series(dtype=float, index=base_idx),
        'regimes': pd.Series(dtype=float, index=base_idx),
        'regime_info': {'means': [np.nan]*nrt, 'variances': [np.nan]*nrt, 'weights': [np.nan]*nrt, 'mapping': {}},
        'actual_num_regimes': 0, 
        'transition_counts': np.zeros((0, 0)), # Will be resized based on observed
        'observed_regime_labels_for_bayes': np.array([]), # Store the GMM labels corresponding to Bayesian analysis indices
        'regime_observations': np.array([]),      
        'transition_probabilities': np.array([]), 
        'credible_intervals': {},
        'probability_mass_functions': {},
        'cumulative_distribution_functions': {},
        'current_regime': None, 
        'regime_return_statistics': [{} for _ in range(nrt)] 
    }


def analyze_stock_with_gmm(symbol: str, data: pd.DataFrame,
                           volatility_method: str = config.DEFAULT_VOLATILITY_METHOD,
                           window: int = config.DEFAULT_VOLATILITY_WINDOW, 
                           num_regimes_target: int = config.DEFAULT_NUM_REGIMES,
                           garch_p: int = config.DEFAULT_GARCH_P, garch_q: int = config.DEFAULT_GARCH_Q,
                           ewma_decay: float = config.DEFAULT_EWMA_DECAY,
                           nn_lookback: int = config.DEFAULT_NN_LOOKBACK,
                           prior_type: str = config.DEFAULT_PRIOR_TYPE,
                           min_prior: float = config.DEFAULT_MIN_PRIOR,
                           max_prior: float = config.DEFAULT_MAX_PRIOR,
                           start_date: str = None, end_date: str = None,
                           random_state_gmm: int = 42) -> dict:
    if data is None or data.empty: return None 

    df_to_analyze = data.copy()
    if start_date or end_date:
        s_d = start_date if start_date else (df_to_analyze.index.min().strftime('%Y-%m-%d') if hasattr(df_to_analyze.index, 'min') and not df_to_analyze.empty else None)
        e_d = end_date if end_date else (df_to_analyze.index.max().strftime('%Y-%m-%d') if hasattr(df_to_analyze.index, 'max') and not df_to_analyze.empty else None)
        if s_d and e_d : df_to_analyze = filter_data_by_date(df_to_analyze, s_d, e_d)

    results = _get_empty_results_template(symbol, df_to_analyze if not df_to_analyze.empty else data, num_regimes_target)
    if df_to_analyze.empty: return results 

    ohlc_cols = ['open', 'high', 'low', 'close']
    results['ohlc_data_filtered'] = df_to_analyze[ohlc_cols].copy() if all(c in df_to_analyze.columns for c in ohlc_cols) else pd.DataFrame(index=df_to_analyze.index, columns=ohlc_cols)

    vol_func_args = {'df': df_to_analyze.copy()} 
    if volatility_method == 'simple':
        vol_func_args['window'] = window
        vol = calculate_simple_volatility(**vol_func_args)
    elif volatility_method == 'garch':
        vol_func_args.update({'p': garch_p, 'q': garch_q, 'fallback_window': window})
        vol = calculate_garch_volatility(**vol_func_args)
    elif volatility_method == 'ewma':
        vol_func_args.update({'decay': ewma_decay, 'fallback_window': window})
        vol = calculate_ewma_volatility(**vol_func_args)
    elif volatility_method == 'sv':
        vol_func_args.update({'fallback_window': window})
        vol = calculate_sv_volatility(**vol_func_args)
    elif volatility_method == 'nn':
        vol_func_args.update({'lookback': nn_lookback, 'fallback_window': window})
        vol = calculate_nn_volatility(**vol_func_args)
    elif volatility_method == 'hybrid':
        vol_func_args.update({'fallback_window': window}) 
        vol = calculate_hybrid_volatility(**vol_func_args)
    else:
        print(f"Unknown volatility method: {volatility_method}. Defaulting to simple.")
        vol_func_args['window'] = window
        vol = calculate_simple_volatility(**vol_func_args)
    
    results['volatility'] = vol if vol is not None else pd.Series(dtype=float, index=df_to_analyze.index)
    if vol is None or vol.dropna().empty: return results

    regimes_ts, gmm_info = identify_volatility_regimes_gmm(vol, num_regimes_target, random_state_gmm)
    results.update({'regimes': regimes_ts, 'regime_info': gmm_info})
    
    actual_n_reg_found = sum(1 for m_idx, m_val in enumerate(gmm_info.get('means', [])) 
                             if not pd.isna(m_val) and m_idx < num_regimes_target and 
                             (not isinstance(m_val, (list, np.ndarray)) or not pd.isna(m_val[0])))

    results['actual_num_regimes'] = actual_n_reg_found
    if actual_n_reg_found == 0: return results
    
    tc_raw_for_observed, observed_regime_labels = count_transitions_improved(regimes_ts)
    results['observed_regime_labels_for_bayes'] = observed_regime_labels # Store this mapping

    if tc_raw_for_observed.shape[0] == 0: # No transitions observed
        # Fill relevant parts of results with empty/NaN based on actual_n_reg_found if needed
        # For now, if no transitions, Bayesian parts will be empty from bayesian_transition_analysis_improved
        pass

    regime_obs_for_bayes = tc_raw_for_observed.sum(axis=1) 

    mean_probs, posterior_s, cred_intervals = bayesian_transition_analysis_improved(
        tc_raw_for_observed, regime_obs_for_bayes, prior_type, min_prior, max_prior
    )
    
    results['transition_counts'] = tc_raw_for_observed 
    results['regime_observations'] = regime_obs_for_bayes 
    results['transition_probabilities'] = mean_probs
    results['credible_intervals'] = cred_intervals 

    pmfs, cdfs = calculate_transition_probability_distributions(posterior_s, cred_intervals)
    results['probability_mass_functions'] = pmfs
    results['cumulative_distribution_functions'] = cdfs
    
    last_reg_val = regimes_ts.dropna().iloc[-1] if not regimes_ts.dropna().empty else None
    if last_reg_val is not None and not pd.isna(last_reg_val):
        current_reg_original_label = int(last_reg_val)
        results['current_regime'] = current_reg_original_label # Store the GMM assigned label
    
    results['regime_return_statistics'] = calculate_regime_return_statistics(
        df_to_analyze, regimes_ts, actual_n_reg_found
    )
    return results