# excel_reporter.py
"""
Generates an Excel report for a single stock's analysis results.
Includes sheets for time series, GMM params, transitions, CIs, return stats, and PMFs.
Now with embedded plots.
"""
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image as OpenpyxlImage # Rename to avoid conflict with PIL if used elsewhere

# Helper to add plot to sheet
def _add_plot_to_sheet(worksheet, plot_buffer, cell_location="A1", chart_width=None, chart_height=None):
    """Adds a plot from a buffer to the specified worksheet cell."""
    try:
        plot_buffer.seek(0)
        img = OpenpyxlImage(plot_buffer)
        if chart_width: img.width = chart_width
        if chart_height: img.height = chart_height
        worksheet.add_image(img, cell_location)
    except Exception as e:
        print(f"Error adding plot to sheet: {e}")
        # Fallback: attempt to save plot to disk for debugging if embedding fails
        try:
            plot_buffer.seek(0)
            debug_plot_path = os.path.join(os.getcwd(), f"debug_plot_error_{worksheet.title.replace(' ', '_')}.png")
            with open(debug_plot_path, 'wb') as f_debug:
                f_debug.write(plot_buffer.read())
            print(f"Debug plot saved to: {debug_plot_path}")
        except Exception as e_debug:
            print(f"Could not save debug plot: {e_debug}")


# Helper sheet creation functions (dataframes remain the same)
def _create_timeseries_sheet_df(results_dict: dict) -> pd.DataFrame:
    # Use index from OHLC filtered data or fallback to volatility index
    base_idx = results_dict.get('ohlc_data_filtered', pd.DataFrame()).index
    if base_idx.empty and results_dict.get('volatility') is not None:
        base_idx = results_dict.get('volatility').index
    
    df_ts = pd.DataFrame(index=base_idx)
    
    ohlc_df = results_dict.get('ohlc_data_filtered')
    if ohlc_df is not None and not ohlc_df.empty:
        for col in ['open', 'high', 'low', 'close']: # Original column names in ohlc_df
            if col in ohlc_df.columns: df_ts[col.capitalize()] = ohlc_df[col] # Capitalized for sheet
    
    vol_s = results_dict.get('volatility')
    if vol_s is not None and not vol_s.empty: df_ts['Volatility'] = vol_s
    
    reg_s = results_dict.get('regimes')
    if reg_s is not None and not reg_s.empty: df_ts['Identified_Regime'] = reg_s
    
    if isinstance(df_ts.index, pd.DatetimeIndex): df_ts.index.name = 'Date'
    return df_ts

def _create_regime_params_sheet_df(results_dict: dict) -> pd.DataFrame:
    reg_info = results_dict.get('regime_info', {})
    # Use actual_num_regimes if available, else from length of means if it's populated
    actual_n_reg = results_dict.get('actual_num_regimes', 0)
    if actual_n_reg == 0 and reg_info.get('means'):
        actual_n_reg = sum(1 for m in reg_info['means'] if not pd.isna(m) and not (isinstance(m, list) and pd.isna(m[0])) )
        
    params_data = []
    if actual_n_reg > 0 and reg_info.get('means'):
        means_list = reg_info.get('means', [])
        variances_list = reg_info.get('variances', [])
        weights_list = reg_info.get('weights', [])

        for i in range(actual_n_reg): # Iterate up to actual GMM components
            mean_vol = means_list[i][0] if isinstance(means_list[i], (list, np.ndarray)) else means_list[i] # Handle 1D/2D means
            var_val = variances_list[i][0][0] if isinstance(variances_list[i], (list,np.ndarray)) and np.array(variances_list[i]).ndim==2 else variances_list[i] # Handle 1D/2D GMM
            std_dev = np.sqrt(var_val) if not pd.isna(var_val) and var_val >= 0 else np.nan
            weight_val = weights_list[i] if i < len(weights_list) else np.nan
            params_data.append({
                'Regime': i, 'Mean_Volatility': mean_vol,
                'StdDev_Volatility': std_dev, 'GMM_Weight': weight_val
            })
    return pd.DataFrame(params_data)

def _create_transitions_sheet_df(results_dict: dict) -> pd.DataFrame:
    current_r_label = results_dict.get('current_regime') # This is GMM label (0,1,2...)
    trans_p_matrix = results_dict.get('transition_probabilities') # (k x k) matrix for observed
    cred_intervals_dict = results_dict.get('credible_intervals') # keys are 0..k-1 for observed
    
    trans_data = []
    num_potential_to_regimes = 0
    from_regime_idx_for_report = None

    if trans_p_matrix is not None and trans_p_matrix.ndim == 2:
        num_potential_to_regimes = trans_p_matrix.shape[1]
        if current_r_label is not None and 0 <= current_r_label < trans_p_matrix.shape[0]:
            # Check if this label (which is an original GMM label) has CIs computed.
            # CI keys are 0..k-1 mapped from observed.
            # This requires mapping current_r_label to its index in observed_regime_labels if it exists.
            # For now, we assume 'current_regime' is directly usable as a key in cred_intervals_dict
            # if it was an observed "from" regime.
            # stock_analyzer stores current_regime as original GMM label.
            # cred_intervals_dict keys are dense 0..k-1.
            # A more robust way:
            # observed_labels = results_dict.get('observed_regime_labels_for_bayes', np.array([]))
            # if current_r_label in observed_labels:
            #    from_regime_idx_for_report = np.where(observed_labels == current_r_label)[0][0]
            # For simplicity here, if current_r_label is a key in cred_intervals_dict, we use it.
            # This implies current_r_label was one of the observed regimes that initiated transitions.
            if current_r_label in cred_intervals_dict: # If current_regime is a key in CIs (0..k-1 space)
                from_regime_idx_for_report = current_r_label
        
    if from_regime_idx_for_report is not None:
        # Ensure from_regime_idx_for_report is valid for trans_p_matrix
        if 0 <= from_regime_idx_for_report < trans_p_matrix.shape[0]:
            probs_row = trans_p_matrix[from_regime_idx_for_report]
            cis_for_row = cred_intervals_dict.get(from_regime_idx_for_report, {})
            lower_cis = cis_for_row.get('lower', np.full(num_potential_to_regimes, np.nan))
            upper_cis = cis_for_row.get('upper', np.full(num_potential_to_regimes, np.nan))

            # The 'To_Regime' in the output df should ideally be the original GMM labels
            # if observed_regime_labels were available and used to map back.
            # For now, to_idx is 0..k-1.
            observed_labels = results_dict.get('observed_regime_labels_for_bayes', np.arange(num_potential_to_regimes))

            for to_idx_dense in range(num_potential_to_regimes): # Iterate over cols of prob matrix (dense 0..k-1)
                to_regime_original_label = observed_labels[to_idx_dense] if to_idx_dense < len(observed_labels) else to_idx_dense

                trans_data.append({
                    'From_Regime': from_regime_idx_for_report, # This is the dense index used for CIs
                    'To_Regime': to_regime_original_label, # This should be the original GMM label
                    'Mean_Probability': probs_row[to_idx_dense] if to_idx_dense < len(probs_row) else np.nan,
                    '95%_CI_Lower': lower_cis[to_idx_dense] if to_idx_dense < len(lower_cis) else np.nan,
                    '95%_CI_Upper': upper_cis[to_idx_dense] if to_idx_dense < len(upper_cis) else np.nan,
                })
    return pd.DataFrame(trans_data)


def _create_regime_stats_sheet_df(results_dict: dict) -> pd.DataFrame:
    stats_list = results_dict.get('regime_return_statistics', [])
    if stats_list and isinstance(stats_list[0], dict): # Ensure it's a list of dicts
        return pd.DataFrame(stats_list)
    return pd.DataFrame()

def _create_pmf_sheet_df(results_dict: dict) -> pd.DataFrame:
    pmfs = results_dict.get('probability_mass_functions', {})
    current_r_label = results_dict.get('current_regime') # GMM label
    
    pmf_data_list = []
    from_regime_idx_for_pmf = None

    # Similar logic to transitions: current_r_label is original GMM label.
    # PMF keys are dense 0..k-1.
    if current_r_label is not None and current_r_label in pmfs: # Check if current_r_label is a key in pmfs
        from_regime_idx_for_pmf = current_r_label

    if from_regime_idx_for_pmf is not None:
        current_regime_pmf_data = pmfs.get(from_regime_idx_for_pmf, {})
        
        # Get observed labels to map To_Regime back to original GMM labels
        num_potential_to_regimes = results_dict.get('transition_probabilities', np.array([])).shape[1]
        observed_labels = results_dict.get('observed_regime_labels_for_bayes', np.arange(num_potential_to_regimes))

        for to_regime_idx_dense, dist_info in current_regime_pmf_data.items(): # to_regime_idx_dense is 0..k-1
            to_regime_original_label = observed_labels[to_regime_idx_dense] if to_regime_idx_dense < len(observed_labels) else to_regime_idx_dense
            if 'bins' in dist_info and 'density' in dist_info:
                is_first_bin = True
                for bin_val, density_val in zip(dist_info['bins'], dist_info['density']):
                    entry = {
                        'From_Regime': from_regime_idx_for_pmf, # This is the dense index
                        'To_Regime': to_regime_original_label, # Original GMM label
                        'Probability_Bin_Center': bin_val, 'Density': density_val
                    }
                    if is_first_bin: 
                        entry['Mean_Prob_Dist'] = dist_info.get('mean', np.nan)
                        entry['Median_Prob_Dist'] = dist_info.get('median', np.nan)
                        entry['StdDev_Prob_Dist'] = dist_info.get('std_dev', np.nan)
                        is_first_bin = False
                    pmf_data_list.append(entry)
    return pd.DataFrame(pmf_data_list)

def save_stock_excel_report_improved(results: dict, symbol: str, save_dir: str):
    if results is None: return
    output_path = os.path.join(save_dir, f"{symbol}_analysis_results_v4.xlsx")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Time Series
            df_ts = _create_timeseries_sheet_df(results)
            sheet1_name = 'Volatility_Regime_TS'
            if not df_ts.empty: 
                df_ts.to_excel(writer, sheet_name=sheet1_name) # Index (Date) is written
                try:
                    fig_ts, ax_ts = plt.subplots(figsize=(14, 8)) # Larger plot
                    # Primary Y-axis for Price
                    if 'Close' in df_ts.columns and not df_ts['Close'].dropna().empty:
                        ax_ts.plot(df_ts.index, df_ts['Close'], label='Close Price', color='blue', alpha=0.7)
                        ax_ts.set_ylabel('Price', color='blue')
                        ax_ts.tick_params(axis='y', labelcolor='blue')
                        ax2_ts = ax_ts.twinx() # Secondary Y-axis for Volatility
                    else: # No 'Close', Volatility on primary
                        ax2_ts = ax_ts 
                    
                    # Plot Volatility on ax2_ts (either primary or secondary)
                    if 'Volatility' in df_ts.columns and not df_ts['Volatility'].dropna().empty:
                        ax2_ts.plot(df_ts.index, df_ts['Volatility'], label='Volatility', color='orange', alpha=0.9)
                        ax2_ts.set_ylabel('Volatility', color='orange')
                        ax2_ts.tick_params(axis='y', labelcolor='orange')

                    # Regime Shading on ax2_ts
                    if 'Identified_Regime' in df_ts.columns and not df_ts['Identified_Regime'].dropna().empty:
                        unique_regimes = sorted(df_ts['Identified_Regime'].dropna().unique().astype(int))
                        cmap_name = 'viridis'
                        colors_regime = plt.cm.get_cmap(cmap_name, len(unique_regimes) if len(unique_regimes) > 0 else 1)
                        
                        min_val_shade, max_val_shade = ax2_ts.get_ylim()
                        if not (np.isfinite(min_val_shade) and np.isfinite(max_val_shade)): # Handle empty ylim
                             min_val_shade, max_val_shade = (0,1) if df_ts['Volatility'].dropna().empty else (df_ts['Volatility'].min(), df_ts['Volatility'].max())


                        for i, regime_val in enumerate(unique_regimes):
                            color_idx = i / (len(unique_regimes) -1 if len(unique_regimes) > 1 else 1)
                            ax2_ts.fill_between(df_ts.index, min_val_shade, max_val_shade, 
                                                where=(df_ts['Identified_Regime'] == regime_val),
                                                facecolor=colors_regime(color_idx), alpha=0.15,
                                                label=f'Regime {regime_val}' if f'Regime {regime_val}' not in [h.get_label() for h in ax2_ts.get_legend_handles_labels()[0]] else "")
                    
                    fig_ts.suptitle(f'{symbol} - Time Series: Price, Volatility & Regimes', fontsize=14)
                    # Consolidate legends
                    handles, labels = [], []
                    for ax_ in [ax_ts, ax2_ts]:
                        if ax_ is not None:
                            h, l = ax_.get_legend_handles_labels()
                            handles.extend(h); labels.extend(l)
                    if handles: fig_ts.legend(handles, labels, loc='upper left', fontsize='small')

                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle and legend
                    
                    ts_plot_buffer = io.BytesIO()
                    fig_ts.savefig(ts_plot_buffer, format='png', dpi=100)
                    plt.close(fig_ts)
                    
                    ws_ts = writer.sheets[sheet1_name]
                    plot_start_row_ts = len(df_ts) + 5 # Rows used by data + header + index col
                    _add_plot_to_sheet(ws_ts, ts_plot_buffer, f'A{plot_start_row_ts}', chart_width=800, chart_height=480)
                except Exception as e_plot:
                    print(f"Error generating time series plot for {symbol}: {e_plot}")
            else: 
                pd.DataFrame([{"Info": "Time series data N/A."}]).to_excel(writer, sheet_name=sheet1_name, index=False)
            
            # Sheet 2: Regime Parameters
            df_params = _create_regime_params_sheet_df(results)
            sheet2_name = 'Regime_Parameters'
            if not df_params.empty: 
                df_params.to_excel(writer, sheet_name=sheet2_name, index=False)
                try:
                    fig_params, ax_params = plt.subplots(figsize=(10, 6))
                    bar_width = 0.35
                    index_regimes = np.arange(len(df_params['Regime']))

                    ax_params.bar(index_regimes - bar_width/2, df_params['Mean_Volatility'], bar_width, label='Mean Volatility', color='skyblue')
                    if 'StdDev_Volatility' in df_params.columns:
                         ax_params.bar(index_regimes + bar_width/2, df_params['StdDev_Volatility'], bar_width, label='StdDev Volatility', color='lightcoral')
                    
                    ax_params.set_xlabel('Regime Label (GMM Sorted by Mean Vol)')
                    ax_params.set_ylabel('Volatility Value')
                    ax_params.set_title(f'{symbol} - GMM Regime Parameters', fontsize=14)
                    ax_params.set_xticks(index_regimes)
                    ax_params.set_xticklabels(df_params['Regime'].astype(int))
                    ax_params.legend()
                    plt.tight_layout()

                    params_plot_buffer = io.BytesIO()
                    fig_params.savefig(params_plot_buffer, format='png', dpi=100)
                    plt.close(fig_params)

                    ws_params = writer.sheets[sheet2_name]
                    plot_start_row_params = len(df_params) + 5
                    _add_plot_to_sheet(ws_params, params_plot_buffer, f'A{plot_start_row_params}', chart_width=700, chart_height=420)
                except Exception as e_plot:
                    print(f"Error generating regime parameters plot for {symbol}: {e_plot}")
            else: 
                pd.DataFrame([{"Info": "GMM parameters N/A."}]).to_excel(writer, sheet_name=sheet2_name, index=False)

            # Sheet 3: Current Regime Transitions
            df_trans = _create_transitions_sheet_df(results)
            sheet3_name = 'Current_Regime_Transitions'
            header_info = {
                "Current_Regime_Label": results.get('current_regime', "N/A"), # This is original GMM Label
                "Analysis_Date": results.get('volatility', pd.Series(dtype=object)).index[-1].strftime('%Y-%m-%d') 
                                 if results.get('volatility') is not None and not results.get('volatility', pd.Series(dtype=object)).empty and isinstance(results.get('volatility').index, pd.DatetimeIndex) else "N/A",
                "Current_Volatility_Value": results.get('volatility', pd.Series(dtype=object)).iloc[-1] 
                                         if results.get('volatility') is not None and not results.get('volatility', pd.Series(dtype=object)).empty else np.nan
            }
            pd.DataFrame([header_info]).to_excel(writer, sheet_name=sheet3_name, index=False, startrow=0)
            
            if not df_trans.empty: 
                df_trans.to_excel(writer, sheet_name=sheet3_name, index=False, startrow=2)
                try:
                    current_regime_label_for_plot = header_info.get("Current_Regime_Label", "N/A")
                    from_regime_dense_idx = df_trans['From_Regime'].iloc[0] # This is the dense index used in CIs

                    fig_trans, ax_trans = plt.subplots(figsize=(10, 6))
                    to_regime_labels_plot = df_trans['To_Regime'].astype(str) # Original GMM labels
                    
                    # Calculate error bar lengths
                    lower_err = df_trans['Mean_Probability'] - df_trans['95%_CI_Lower']
                    upper_err = df_trans['95%_CI_Upper'] - df_trans['Mean_Probability']
                    
                    ax_trans.bar(to_regime_labels_plot, df_trans['Mean_Probability'],
                                 yerr=[lower_err.values, upper_err.values], # Pass as array/list
                                 capsize=5, label='Mean Transition Probability w/ 95% CI', color='mediumseagreen')
                    
                    ax_trans.set_xlabel('To Regime (Original GMM Label)')
                    ax_trans.set_ylabel('Probability')
                    ax_trans.set_title(f'{symbol} - Transitions from Current Regime {current_regime_label_for_plot} (mapped to dense index {from_regime_dense_idx} for CIs)', fontsize=12)
                    ax_trans.legend()
                    plt.ylim(0, max(1.05, df_trans['95%_CI_Upper'].max() * 1.1 if not df_trans['95%_CI_Upper'].empty else 1.05) ) # Adjust ylim for CIs
                    plt.tight_layout()

                    trans_plot_buffer = io.BytesIO()
                    fig_trans.savefig(trans_plot_buffer, format='png', dpi=100)
                    plt.close(fig_trans)

                    ws_trans = writer.sheets[sheet3_name]
                    plot_start_row_trans = len(df_trans) + 2 + 5 # +2 for header, +5 for spacing
                    _add_plot_to_sheet(ws_trans, trans_plot_buffer, f'A{plot_start_row_trans}', chart_width=700, chart_height=420)
                except Exception as e_plot:
                    print(f"Error generating transitions plot for {symbol}: {e_plot}")
            else: 
                pd.DataFrame([{"Info": "Transition data N/A."}]).to_excel(writer, sheet_name=sheet3_name, index=False, startrow=2)
            
            # Sheet 4: Regime Return Statistics
            df_ret_stats = _create_regime_stats_sheet_df(results)
            sheet4_name = 'Regime_Return_Stats'
            if not df_ret_stats.empty: 
                df_ret_stats.to_excel(writer, sheet_name=sheet4_name, index=False)
                # Optional: Add a plot for Sharpe Ratios or Mean Returns per regime if desired
            else: 
                pd.DataFrame([{"Info": "Return stats N/A."}]).to_excel(writer, sheet_name=sheet4_name, index=False)
            
            # Sheet 5: Probability Distributions (PMFs for transitions from current regime)
            df_pmf = _create_pmf_sheet_df(results)
            sheet5_name = 'Probability_Distributions'
            if not df_pmf.empty: 
                df_pmf.to_excel(writer, sheet_name=sheet5_name, index=False)
                try:
                    from_regime_dense_idx_pmf = df_pmf['From_Regime'].iloc[0] if 'From_Regime' in df_pmf.columns and not df_pmf.empty else None
                    current_actual_gmm_label = results.get('current_regime', 'N/A')

                    if from_regime_dense_idx_pmf is not None:
                        unique_to_regimes_for_pmf = sorted(df_pmf['To_Regime'].unique()) # Original GMM labels
                        num_pmf_plots = len(unique_to_regimes_for_pmf)

                        if num_pmf_plots > 0:
                            cols_subplot = min(num_pmf_plots, 3) 
                            rows_subplot = (num_pmf_plots + cols_subplot - 1) // cols_subplot
                            
                            fig_pmf, axes_pmf = plt.subplots(rows_subplot, cols_subplot, 
                                                             figsize=(6 * cols_subplot, 5 * rows_subplot), squeeze=False)
                            axes_pmf_flat = axes_pmf.flatten()
                            plot_idx = 0
                            for to_reg_original_label in unique_to_regimes_for_pmf:
                                pmf_subset = df_pmf[(df_pmf['From_Regime'] == from_regime_dense_idx_pmf) & 
                                                    (df_pmf['To_Regime'] == to_reg_original_label)]
                                if not pmf_subset.empty and plot_idx < len(axes_pmf_flat):
                                    ax = axes_pmf_flat[plot_idx]
                                    # Calculate bin width (approximate)
                                    bin_centers = pmf_subset['Probability_Bin_Center']
                                    bin_width = (bin_centers.iloc[1] - bin_centers.iloc[0]) if len(bin_centers) > 1 else 0.05
                                    
                                    ax.bar(bin_centers, pmf_subset['Density'], width=bin_width * 0.9, label='Density', color='lightsteelblue')
                                    mean_prob_dist = pmf_subset['Mean_Prob_Dist'].iloc[0]
                                    if not pd.isna(mean_prob_dist):
                                        ax.axvline(mean_prob_dist, color='r', linestyle='--', label=f'Mean: {mean_prob_dist:.3f}')
                                    
                                    ax.set_title(f'PMF: From Current ({current_actual_gmm_label}, dense {from_regime_dense_idx_pmf}) to {to_reg_original_label}', fontsize=10)
                                    ax.set_xlabel('Transition Probability Bin')
                                    ax.set_ylabel('Density')
                                    ax.legend(fontsize='small')
                                    ax.tick_params(axis='x', labelsize=8)
                                    ax.tick_params(axis='y', labelsize=8)
                                    plot_idx += 1
                            
                            for i in range(plot_idx, len(axes_pmf_flat)): fig_pmf.delaxes(axes_pmf_flat[i])

                            fig_pmf.suptitle(f'{symbol} - Transition Probability Distributions from Current Regime ({current_actual_gmm_label}, dense index {from_regime_dense_idx_pmf})', fontsize=14)
                            plt.tight_layout(rect=[0, 0, 1, 0.95])

                            pmf_plot_buffer = io.BytesIO()
                            fig_pmf.savefig(pmf_plot_buffer, format='png', dpi=100)
                            plt.close(fig_pmf)

                            ws_pmf = writer.sheets[sheet5_name]
                            plot_start_row_pmf = len(df_pmf) + 5
                            _add_plot_to_sheet(ws_pmf, pmf_plot_buffer, f'A{plot_start_row_pmf}', chart_width=750 * cols_subplot // 2, chart_height=450 * rows_subplot //2) # Adjust size
                except Exception as e_plot:
                    print(f"Error generating PMF plots for {symbol}: {e_plot}")
            else: 
                pd.DataFrame([{"Info": "Probability distribution data N/A."}]).to_excel(writer, sheet_name=sheet5_name, index=False)

    except Exception as e:
        print(f"Error saving Excel report for {symbol} to {output_path}: {e}")