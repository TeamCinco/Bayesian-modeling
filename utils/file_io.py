import os
import csv
import json
import numpy as np
from datetime import datetime

def save_results(results, output_dir, ticker='SPY'):
    """
    Save analysis results to CSV files and numpy arrays
    
    Parameters:
    -----------
    results : dict
        Analysis results from analyze_market_regimes
    output_dir : str
        Directory to save output files
    ticker : str
        Ticker symbol for the analyzed asset
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save regime assignments
    regime_file = os.path.join(output_dir, f"{ticker}_regimes.csv")
    with open(regime_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Close', 'Return', 'Volatility', 'ATR', 'Regime', 'Regime1_Prob', 'Regime2_Prob', 'Regime3_Prob'])
        
        for i in range(len(results['dates'])):
            # Skip first row (no return calculated)
            if i == 0:
                continue
                
            row = [
                results['dates'][i],
                results['ohlcv']['close'][i],
                results['features']['log_returns'][i-1],  # Offset because returns start at index 1
                results['features'].get('volatility', [0] * len(results['dates']))[i-1] if 'volatility' in results['features'] and i-1 < len(results['features']['volatility']) else 0,
                results['features'].get('atr', [0] * len(results['dates']))[i] if 'atr' in results['features'] and i < len(results['features'].get('atr', [])) else 0,
                results['regimes'][i-1] + 1 if i-1 < len(results['regimes']) else 0,  # 1-indexed for readability
            ]
            
            # Add regime probabilities if available
            if 'regime_probs' in results and i-1 < len(results['regime_probs']):
                for p in results['regime_probs'][i-1]:
                    row.append(f"{p:.4f}")
            else:
                # Add placeholders if regime_probs not available
                for _ in range(results.get('hmm_model', {}).get('n_regimes', 3)):
                    row.append("0.0000")
                
            writer.writerow(row)
    
    # Save regime statistics
    stats_file = os.path.join(output_dir, f"{ticker}_regime_stats.csv")
    with open(stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Regime', 'Count', 'Percentage', 'Mean_Return', 'Volatility', 
                        'Sharpe', 'Mean_ATR', 'Start_Date', 'End_Date', 'Description'])
        
        for stat in sorted(results['regime_stats'], key=lambda x: x['regime']):
            writer.writerow([
                stat['regime'] + 1,  # 1-indexed for readability
                stat['count'],
                f"{stat['percentage']:.2f}",
                f"{stat['mean_return']*100:.4f}",
                f"{stat['volatility']*100:.4f}",
                f"{stat.get('sharpe', 0):.4f}",  # Use .get() with default value
                f"{stat.get('mean_atr', 0.01):.6f}",  # Use .get() with default value
                stat.get('start_date', ''),
                stat.get('end_date', ''),
                stat.get('description', '')
            ])
    
    # Save model parameters - check which type of model is used
    model_file = os.path.join(output_dir, f"{ticker}_model_params.csv")
    with open(model_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        
        # Save number of regimes
        writer.writerow(['n_regimes', results['hmm_model'].n_regimes])
        
        # Check model type and save appropriate parameters
        if hasattr(results['hmm_model'], 'mu') and hasattr(results['hmm_model'], 'sigma'):
            # It's a BayesianHMM model
            writer.writerow(['model_type', 'BayesianHMM'])
            
            # Save mean parameters
            for i, mu in enumerate(results['hmm_model'].mu):
                writer.writerow([f'mu_regime_{i+1}', f"{mu:.6f}"])
            
            # Save volatility parameters
            for i, sigma in enumerate(results['hmm_model'].sigma):
                writer.writerow([f'sigma_regime_{i+1}', f"{sigma:.6f}"])
            
            # Save transition matrix if it exists
            if hasattr(results['hmm_model'], 'trans_mat') or hasattr(results['hmm_model'], 'transmat_'):
                writer.writerow(['transition_matrix', ''])
                trans_mat = getattr(results['hmm_model'], 'trans_mat', None)
                if trans_mat is None:
                    trans_mat = getattr(results['hmm_model'], 'transmat_', [])
                
                for i, row in enumerate(trans_mat):
                    writer.writerow([f'trans_from_regime_{i+1}', ','.join(f"{p:.4f}" for p in row)])
            
            # Save initial probabilities if they exist
            if hasattr(results['hmm_model'], 'init_probs') or hasattr(results['hmm_model'], 'startprob_'):
                init_probs = getattr(results['hmm_model'], 'init_probs', None)
                if init_probs is None:
                    init_probs = getattr(results['hmm_model'], 'startprob_', [])
                
                writer.writerow(['initial_probs', ','.join(f"{p:.4f}" for p in init_probs)])
        
        elif hasattr(results['hmm_model'], 'gmm'):
            # It's a BayesianRegimePersistence model
            writer.writerow(['model_type', 'BayesianRegimePersistence'])
            
            # Save GMM parameters if available
            if results['hmm_model'].gmm is not None:
                writer.writerow(['gmm_means', ','.join(f"{m[0]:.6f}" for m in results['hmm_model'].gmm.means_)])
                writer.writerow(['gmm_covariances', ','.join(f"{c[0][0]:.6f}" for c in results['hmm_model'].gmm.covariances_)])
                writer.writerow(['gmm_weights', ','.join(f"{w:.4f}" for w in results['hmm_model'].gmm.weights_)])
            
            # Save feature names
            if hasattr(results['hmm_model'], 'feature_names') and results['hmm_model'].feature_names:
                writer.writerow(['feature_names', ','.join(results['hmm_model'].feature_names)])
        
        else:
            # Unknown model type
            writer.writerow(['model_type', 'Unknown'])
    
    # Save extended statistics if available
    if 'extended_stats' in results:
        # Save as CSV
        extended_stats_file = os.path.join(output_dir, f"{ticker}_extended_stats.csv")
        
        # Get all keys across all regime stats
        all_keys = set()
        for stat in results['extended_stats']:
            all_keys.update(stat.keys())
        
        # Remove 'durations' as it's a list and not easily saved to CSV
        if 'durations' in all_keys:
            all_keys.remove('durations')
        
        with open(extended_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['Regime'] + list(all_keys)
            writer.writerow(header)
            
            # Write data for each regime
            for stat in sorted(results['extended_stats'], key=lambda x: x['regime']):
                row = [stat['regime'] + 1]  # 1-indexed for readability
                
                for key in all_keys:
                    if key in stat:
                        value = stat[key]
                        # Format based on type
                        if isinstance(value, float):
                            formatted_value = f"{value:.6f}"
                        else:
                            formatted_value = str(value)
                        row.append(formatted_value)
                    else:
                        row.append('')
                        
                writer.writerow(row)
    
    # Save risk metrics if available
    if 'risk_metrics' in results:
        risk_metrics_file = os.path.join(output_dir, f"{ticker}_risk_metrics.csv")
        
        with open(risk_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Get all keys across all metric entries
            all_keys = set()
            for metric in results['risk_metrics']:
                all_keys.update(metric.keys())
            
            # Write header
            header = ['Regime'] + list(all_keys - {'regime'})  # Remove 'regime' as it's already the first column
            writer.writerow(header)
            
            # Write data for each regime
            for metric in sorted(results['risk_metrics'], key=lambda x: x['regime']):
                row = [metric['regime'] + 1]  # 1-indexed for readability
                
                for key in all_keys:
                    if key != 'regime' and key in metric:
                        value = metric[key]
                        # Format based on type
                        if isinstance(value, float):
                            formatted_value = f"{value:.6f}"
                        else:
                            formatted_value = str(value)
                        row.append(formatted_value)
                    elif key != 'regime':
                        row.append('')
                        
                writer.writerow(row)
    
    # Save duration statistics if available
    if 'duration_stats' in results:
        duration_stats_file = os.path.join(output_dir, f"{ticker}_duration_stats.csv")
        
        with open(duration_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Regime', 'Mean_Duration', 'Median_Duration', 'Min_Duration', 
                           'Max_Duration', 'Total_Duration', 'Count'])
            
            # Write data for each regime
            for regime, stats in sorted(results['duration_stats'].items()):
                # Skip occurrences as it's a list and not easily saved to CSV
                row = [
                    regime + 1,  # 1-indexed for readability
                    f"{stats.get('mean_duration', 0):.2f}",
                    f"{stats.get('median_duration', 0):.1f}",
                    stats.get('min_duration', 0),
                    stats.get('max_duration', 0),
                    stats.get('total_duration', 0),
                    stats.get('count', 0)
                ]
                
                writer.writerow(row)
    
    # Save all results as numpy arrays for later use
    np_file = os.path.join(output_dir, f"{ticker}_results.npz")
    
    # Create a dictionary of items to save, checking each one
    save_dict = {
        'dates': np.array(results['dates']),
        'regimes': np.array(results['regimes']),
        'regime_stats': results['regime_stats'],
    }
    
    # Add optional items if they exist
    if 'hmm_model' in results:
        save_dict['model_type'] = np.array([type(results['hmm_model']).__name__])
    if 'regime_probs' in results:
        save_dict['regime_probs'] = np.array(results['regime_probs'])
    if 'regime_mapping' in results:
        save_dict['regime_mapping'] = results['regime_mapping']
    if 'extended_stats' in results:
        save_dict['extended_stats'] = results['extended_stats']
    if 'risk_metrics' in results:
        save_dict['risk_metrics'] = results['risk_metrics']
    if 'duration_stats' in results:
        save_dict['duration_stats'] = results['duration_stats']
    
    np.savez(np_file, **save_dict)
    
    print(f"Results saved to {output_dir}")


def save_cross_asset_analysis(regime_groups, output_dir):
    """
    Save cross-asset regime analysis to CSV
    
    Parameters:
    -----------
    regime_groups : dict
        Dictionary mapping regime indices to lists of tickers
    output_dir : str
        Directory to save output file
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save cross-asset analysis
    cross_asset_file = os.path.join(output_dir, "cross_asset_regimes.csv")
    with open(cross_asset_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Regime', 'Asset_Count', 'Percentage', 'Assets'])
        
        total_assets = sum(len(tickers) for tickers in regime_groups.values())
        
        for regime, ticker_list in sorted(regime_groups.items()):
            percentage = (len(ticker_list) / total_assets) * 100
            writer.writerow([
                regime,
                len(ticker_list),
                f"{percentage:.1f}%",
                ', '.join(ticker_list)
            ])
    
    # Save as JSON for easier programmatic access
    cross_asset_json = os.path.join(output_dir, "cross_asset_regimes.json")
    
    # Convert regime keys to strings for JSON
    json_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'regimes': {str(regime): ticker_list for regime, ticker_list in regime_groups.items()},
        'total_assets': total_assets,
        'percentages': {str(regime): (len(ticker_list) / total_assets) * 100 
                      for regime, ticker_list in regime_groups.items()}
    }
    
    with open(cross_asset_json, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Cross-asset analysis saved to {cross_asset_file} and {cross_asset_json}")


def create_summary_file(output_dir):
    """
    Create a summary file for collecting analysis results
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the summary file
    
    Returns:
    --------
    summary_file : str
        Path to the created summary file
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create summary file
    summary_file = os.path.join(output_dir, "regime_analysis_summary.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ticker', 'Current_Regime', 'Regime_Description', 
                         'Primary_Strategy', 'Alternative_Strategy', 
                         'Position_Sizing', 'Wing_Width', 'Expiration',
                         'Confidence', 'Transition_Risk', 'Expected_Duration',
                         'Stability'])
    
    return summary_file


def add_to_summary(summary_file, ticker, current_regime, recommendations=None, error=None):
    """
    Add analysis results to the summary file
    
    Parameters:
    -----------
    summary_file : str
        Path to the summary file
    ticker : str
        Ticker symbol
    current_regime : int
        Current regime index (or None if error)
    recommendations : dict
        Strategy recommendations (or None if error)
    error : str
        Error message (or None if successful)
    """
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if error:
            # Log error to summary
            writer.writerow([ticker, 'ERROR', str(error), '', '', '', '', '', '', '', '', ''])
        else:
            # Extract stability info if available
            expected_duration = ''
            stability = ''
            
            if recommendations and 'forecasts' in recommendations and 'stability' in recommendations['forecasts']:
                stability_info = recommendations['forecasts']['stability']
                expected_duration = stability_info.get('expected_remaining_duration', '')
                stability = stability_info.get('stability_description', '')
            
            # Add successful analysis to summary
            if recommendations:
                writer.writerow([
                    ticker,
                    current_regime + 1 if current_regime is not None else '',  # 1-indexed for readability
                    recommendations.get('current_regime', ''),
                    recommendations.get('primary_strategy', ''),
                    recommendations.get('alternative_strategy', ''),
                    recommendations.get('position_sizing', ''),
                    recommendations.get('wing_width', ''),
                    recommendations.get('expiration', ''),
                    recommendations.get('confidence', ''),
                    f"{recommendations.get('transition_risk', {}).get('description', '')} ({recommendations.get('transition_risk', {}).get('probability', 0):.2f})" if 'transition_risk' in recommendations else '',
                    expected_duration,
                    stability
                ])
            else:
                # Fallback if recommendations is None but no error was provided
                writer.writerow([ticker, str(current_regime) if current_regime is not None else 'UNKNOWN', '', '', '', '', '', '', '', '', '', ''])