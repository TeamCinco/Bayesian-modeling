# summary_generator.py
"""
Creates and saves an overall summary Excel file from all processed stock results.
Updated to include new metrics from advanced analysis.
"""
import pandas as pd
import numpy as np
import os

def generate_overall_summary(all_calculated_results: dict,
                             num_regimes_target: int, # Target GMM regimes for column consistency
                             save_dir: str,
                             summary_filename: str) -> list:
    summary_list = []
    sorted_symbols = sorted(all_calculated_results.keys())

    for symbol in sorted_symbols:
        res = all_calculated_results.get(symbol)
        if res is None: continue

        # Basic Info
        entry = {'Symbol': symbol,
                 'Current_Regime_Label': res.get('current_regime', 'N/A'), # GMM label
                 'Actual_Num_Regimes_GMM': res.get('actual_num_regimes', 0)}
        vol_s = res.get('volatility')
        entry['Current_Raw_Volatility'] = vol_s.dropna().iloc[-1] if vol_s is not None and not vol_s.dropna().empty else np.nan
        
        # Current Regime GMM parameters
        curr_r_label = res.get('current_regime')
        act_n_r_gmm = res.get('actual_num_regimes',0)
        r_info = res.get('regime_info',{})
        if curr_r_label is not None and 0 <= curr_r_label < act_n_r_gmm and \
           r_info.get('means') and curr_r_label < len(r_info.get('means',[])):
            mean_val = r_info['means'][curr_r_label]
            entry['Current_Regime_Mean_Vol'] = mean_val[0] if isinstance(mean_val, (list,np.ndarray)) else mean_val
        else: entry['Current_Regime_Mean_Vol'] = np.nan

        # Transition probabilities from Current Regime
        # This needs careful handling of current_regime_label vs transition matrix indices
        trans_p_matrix = res.get('transition_probabilities') # (k x k) for observed regimes
        cred_intervals = res.get('credible_intervals')       # keys 0..k-1 for observed regimes
        
        # For summary, assume current_regime_label (if valid GMM label) maps to an index in trans_p_matrix.
        # A robust mapping is needed if label space is sparse or different.
        # Simplified: if current_r_label is a valid index for the trans_p_matrix:
        prob_stay_same = np.nan
        if curr_r_label is not None and trans_p_matrix is not None and \
           0 <= curr_r_label < trans_p_matrix.shape[0] and \
           0 <= curr_r_label < trans_p_matrix.shape[1]: # Check it's a valid cell
            prob_stay_same = trans_p_matrix[curr_r_label, curr_r_label]
        entry['Prob_Stay_Same_Regime'] = prob_stay_same

        for i in range(num_regimes_target): # Columns for Prob_To_Regime_0, _1, _2...
            prob_to_i = np.nan
            if curr_r_label is not None and trans_p_matrix is not None and \
               0 <= curr_r_label < trans_p_matrix.shape[0] and \
               i < trans_p_matrix.shape[1]: # Target regime i is a valid column
                prob_to_i = trans_p_matrix[curr_r_label, i]
            entry[f'Prob_To_Regime_{i}'] = prob_to_i

        # GMM parameters for all target regimes
        for i in range(num_regimes_target):
            mean_i, std_i, weight_i = np.nan, np.nan, np.nan
            if i < act_n_r_gmm and r_info.get('means') and i < len(r_info.get('means',[])):
                gmm_mean_i = r_info['means'][i]
                mean_i = gmm_mean_i[0] if isinstance(gmm_mean_i, (list,np.ndarray)) else gmm_mean_i
                gmm_var_i = r_info.get('variances',[])[i] if i < len(r_info.get('variances',[])) else np.nan
                var_i_val = gmm_var_i[0][0] if isinstance(gmm_var_i,(list,np.ndarray)) and np.array(gmm_var_i).ndim==2 else gmm_var_i
                std_i = np.sqrt(var_i_val) if not pd.isna(var_i_val) and var_i_val >=0 else np.nan
                weight_i = r_info.get('weights',[])[i] if i < len(r_info.get('weights',[])) else np.nan
            entry.update({f'Regime_{i}_Mean_Vol':mean_i, f'Regime_{i}_StdDev_Vol':std_i, f'Regime_{i}_GMM_Weight':weight_i})
        
        # Return Statistics for all target regimes
        ret_stats_list = res.get('regime_return_statistics', [])
        for i in range(num_regimes_target):
            s_i = next((s for s in ret_stats_list if s.get('Regime') == i), None)
            entry[f'Regime_{i}_N_Days'] = s_i.get('N_Days', np.nan) if s_i else np.nan
            entry[f'Regime_{i}_Sharpe'] = s_i.get('Annualized_Sharpe_Ratio', np.nan) if s_i else np.nan
            entry[f'Regime_{i}_Mean_Daily_LogRet'] = s_i.get('Mean_Daily_Log_Return', np.nan) if s_i else np.nan
            entry[f'Regime_{i}_Max_Drawdown'] = s_i.get('Max_Drawdown', np.nan) if s_i else np.nan
        summary_list.append(entry)

    if not summary_list: return []
    summary_df = pd.DataFrame(summary_list)
    if summary_df.empty: return []

    cols_order = ['Symbol', 'Current_Regime_Label', 'Actual_Num_Regimes_GMM', 'Current_Raw_Volatility',
                  'Current_Regime_Mean_Vol', 'Prob_Stay_Same_Regime']
    for i in range(num_regimes_target): cols_order.append(f'Prob_To_Regime_{i}')
    for i in range(num_regimes_target):
        cols_order.extend([f'Regime_{i}_Mean_Vol', f'Regime_{i}_StdDev_Vol', f'Regime_{i}_GMM_Weight',
                           f'Regime_{i}_N_Days', f'Regime_{i}_Sharpe', f'Regime_{i}_Mean_Daily_LogRet', f'Regime_{i}_Max_Drawdown'])
    
    final_cols = [col for col in cols_order if col in summary_df.columns]
    summary_df = summary_df[final_cols].sort_values('Symbol')
    
    summary_path = os.path.join(save_dir, summary_filename)
    try:
        summary_df.to_excel(summary_path, index=False, engine='openpyxl')
        print(f"\nOverall summary (v4) saved to {summary_path}")
    except Exception as e: print(f"Error saving overall summary Excel: {e}")
    return summary_list