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
                results['features']['volatility'][i-1],
                results['features']['atr'][i],
                results['regimes'][i-1] + 1,  # 1-indexed for readability
            ]
            
            # Add regime probabilities
            for p in results['regime_probs'][i-1]:
                row.append(f"{p:.4f}")
                
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
                f"{stat['sharpe']:.4f}",
                f"{stat['mean_atr']:.6f}",
                stat['start_date'],
                stat['end_date'],
                stat['description']
            ])
    
    # Save HMM model parameters
    model_file = os.path.join(output_dir, f"{ticker}_hmm_params.csv")
    with open(model_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        
        # Save number of regimes
        writer.writerow(['n_regimes', results['hmm_model'].n_regimes])
        
        # Save mean parameters
        for i, mu in enumerate(results['hmm_model'].mu):
            writer.writerow([f'mu_regime_{i+1}', f"{mu:.6f}"])
        
        # Save volatility parameters
        for i, sigma in enumerate(results['hmm_model'].sigma):
            writer.writerow([f'sigma_regime_{i+1}', f"{sigma:.6f}"])
        
        # Save transition matrix
        writer.writerow(['transition_matrix', ''])
        for i, row in enumerate(results['hmm_model'].transmat_):
            writer.writerow([f'trans_from_regime_{i+1}', ','.join(f"{p:.4f}" for p in row)])
        
        # Save initial probabilities
        writer.writerow(['initial_probs', ','.join(f"{p:.4f}" for p in results['hmm_model'].startprob_)])
    
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
    np.savez(
        np_file,
        dates=np.array(results['dates']),
        regimes=np.array(results['regimes']),
        regime_stats=results['regime_stats'],
        hmm_model=results['hmm_model'],
        regime_mapping=results.get('regime_mapping', {}),
        regime_probs=np.array(results['regime_probs']),
        extended_stats=results.get('extended_stats', []),
        risk_metrics=results.get('risk_metrics', []),
        duration_stats=results.get('duration_stats', {})
    )
    
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
            
            if 'forecasts' in recommendations and 'stability' in recommendations['forecasts']:
                stability_info = recommendations['forecasts']['stability']
                expected_duration = stability_info.get('expected_remaining_duration', '')
                stability = stability_info.get('stability_description', '')
            
            # Add successful analysis to summary
            writer.writerow([
                ticker,
                current_regime + 1,  # 1-indexed for readability
                recommendations['current_regime'],
                recommendations['primary_strategy'],
                recommendations['alternative_strategy'],
                recommendations['position_sizing'],
                recommendations['wing_width'],
                recommendations['expiration'],
                recommendations['confidence'],
                f"{recommendations['transition_risk']['description']} ({recommendations['transition_risk']['probability']:.2f})",
                expected_duration,
                stability
            ])
