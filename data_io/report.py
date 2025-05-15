"""
Report generation functions for market regime analysis.

This module provides functions to generate formatted text reports
about market regimes and their characteristics.
"""

import os
from datetime import datetime
import textwrap
import numpy as np
import pandas as pd


def generate_regime_report(results, output_path, include_time_series=False):
    """
    Generate a detailed text report of regime analysis results
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary containing regimes, statistics, etc.
    output_path : str
        Path to save the report file
    include_time_series : bool, optional
        Whether to include the full time series data in the report
        
    Returns:
    --------
    output_path : str
        Path to the created report file
    """
    # Extract data
    dates = results['dates']
    regimes = results['regimes']
    regime_stats = results['regime_stats']
    regime_probs = results.get('regime_probs', None)
    hmm = results.get('hmm_model', None)
    
    # Ensure directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("MARKET REGIME ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Analysis information
        f.write("ANALYSIS INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if dates:
            f.write(f"Data range: {dates[0]} to {dates[-1]}\n")
            f.write(f"Number of observations: {len(dates)}\n")
        else:
            f.write("Data range: Not available (no dates provided)\n")
            f.write("Number of observations: 0\n")

        if regimes:
            f.write(f"Number of regimes: {max(regimes) + 1}\n\n")
        else:
            f.write("Number of regimes: Not available (no regimes identified)\n\n")

        # Current regime information
        if regimes and dates:  # Ensure there is data to get current regime
            current_regime_idx = regimes[-1]
            # Check if regime_probs is valid and has data for the last time step
            current_prob = None
            if regime_probs and len(regime_probs) == len(dates) and regime_probs[-1] and current_regime_idx < len(regime_probs[-1]):
                current_prob = regime_probs[-1][current_regime_idx]
            
            current_stat = next((stat for stat in regime_stats if stat['regime'] == current_regime_idx), None)
            
            f.write("CURRENT MARKET REGIME\n")
            f.write("-" * 80 + "\n")
            f.write(f"Current regime: {current_regime_idx}\n")
            if current_prob is not None:
                f.write(f"Confidence: {current_prob:.2f} ({current_prob*100:.1f}%)\n")
            if current_stat:
                f.write(f"Description: {current_stat.get('description', 'N/A')}\n")
                f.write(f"Mean daily return: {current_stat.get('mean_return', 0)*100:.4f}%\n")
                f.write(f"Annualized return: {current_stat.get('mean_return', 0)*252*100:.2f}%\n")
                f.write(f"Volatility (daily): {current_stat.get('volatility', 0)*100:.4f}%\n")
                f.write(f"Volatility (annualized): {current_stat.get('volatility', 0)*np.sqrt(252)*100:.2f}%\n")
                f.write(f"Sharpe ratio: {current_stat.get('sharpe', 0):.4f}\n")
            f.write("\n")
        else:
            f.write("CURRENT MARKET REGIME\n")
            f.write("-" * 80 + "\n")
            f.write("Not available (no regime data).\n\n")
        
        # Regime statistics
        f.write("REGIME STATISTICS\n")
        f.write("-" * 80 + "\n")
        if regime_stats:
            for stat in regime_stats:
                regime_idx = stat['regime']  # Renamed to avoid conflict
                f.write(f"Regime {regime_idx}: {stat.get('description', 'N/A')}\n")
                f.write(f"  Frequency: {stat.get('count', 0)} days ({stat.get('percentage', 0):.1f}%)\n")
                
                if 'start_date' in stat and 'end_date' in stat and stat['start_date'] and stat['end_date']:
                    f.write(f"  First observed: {stat['start_date']}\n")
                    f.write(f"  Last observed: {stat['end_date']}\n")
                
                f.write(f"  Mean daily return: {stat.get('mean_return', 0)*100:.4f}%\n")
                f.write(f"  Annualized return: {stat.get('mean_return', 0)*252*100:.2f}%\n")
                f.write(f"  Volatility (daily): {stat.get('volatility', 0)*100:.4f}%\n")
                f.write(f"  Volatility (annualized): {stat.get('volatility', 0)*np.sqrt(252)*100:.2f}%\n")
                f.write(f"  Sharpe ratio: {stat.get('sharpe', 0):.4f}\n")
                f.write(f"  ATR: {stat.get('mean_atr', 0)*100:.4f}%\n")
                f.write("\n")
        else:
            f.write("No regime statistics available.\n\n")
        
        # Transition probabilities
        if hmm is not None:
            # Get transition matrix with proper error handling
            transition_matrix = None
            if hasattr(hmm, 'transmat_') and hmm.transmat_ is not None:
                transition_matrix = hmm.transmat_
            elif hasattr(hmm, 'trans_mat') and hmm.trans_mat is not None:
                transition_matrix = hmm.trans_mat
            
            if transition_matrix is not None:
                # Convert to numpy array if it's a list
                if isinstance(transition_matrix, list):
                    print(f"Converting transition matrix from list to numpy array for report")
                    try:
                        transition_matrix = np.array(transition_matrix)
                    except Exception as e:
                        print(f"Error converting transition matrix: {str(e)}")
                        transition_matrix = None
                
                if transition_matrix is not None:
                    if not isinstance(transition_matrix, np.ndarray):
                        print(f"Warning: Unexpected transition matrix type: {type(transition_matrix)}")
                    else:
                        f.write("REGIME TRANSITION PROBABILITIES\n")
                        f.write("-" * 80 + "\n")
                        f.write("Probability of transitioning from row regime to column regime:\n\n")
                        
                        n_regimes = transition_matrix.shape[0]
                        
                        # Header row
                        f.write("       |")
                        for j in range(n_regimes):
                            f.write(f" To {j:2d} |")
                        f.write("\n")
                        f.write("-------|" + "-------|" * n_regimes + "\n")
                        
                        # Data rows
                        for i in range(n_regimes):
                            f.write(f"From {i:2d} |")
                            for j in range(n_regimes):
                                f.write(f" {transition_matrix[i][j]:5.2f} |")
                            f.write("\n")
                        f.write("\n")
                        
                        # Highlight most likely transitions
                        f.write("Most likely transitions:\n")
                        for i in range(n_regimes):
                            most_likely = np.argmax(transition_matrix[i])
                            if i != most_likely:  # Skip if most likely to stay in same regime
                                probability = transition_matrix[i][most_likely]
                                # Get descriptions
                                from_desc = next((stat['description'] for stat in regime_stats if stat['regime'] == i), f"Regime {i}")
                                to_desc = next((stat['description'] for stat in regime_stats if stat['regime'] == most_likely), f"Regime {most_likely}")
                                f.write(f"  Regime {i} ({from_desc}) → Regime {most_likely} ({to_desc}): {probability:.2f}\n")
                        f.write("\n")
        
        # Time series data
        if include_time_series:
            f.write("TIME SERIES DATA\n")
            f.write("-" * 80 + "\n")
            if dates and regimes:
                # Make sure lengths match
                min_length = min(len(dates), len(regimes))
                dates_truncated = dates[:min_length]
                regimes_truncated = regimes[:min_length]
                regime_probs_truncated = regime_probs[:min_length] if regime_probs else None
                
                f.write("Date       | Regime | Description")
                if regime_probs_truncated:
                    f.write(" | Probability")
                f.write("\n")
                f.write("-----------|--------|")
                # Adjust width for description; assume max 50 for description
                f.write("-" * 50)
                if regime_probs_truncated:
                    f.write("|------------")
                f.write("\n")
                
                # Only include the last 100 data points to keep the report manageable
                start_idx = max(0, min_length - 100)
                for i in range(start_idx, min_length):
                    date_val = dates_truncated[i]
                    regime_val = regimes_truncated[i]
                    
                    # Get description
                    description = "N/A"
                    if regime_stats:
                        description = next((stat['description'] for stat in regime_stats if stat['regime'] == regime_val), "N/A")
                    # Truncate description if too long
                    if len(description) > 47:  # Adjusted for "..."
                        description = description[:47] + "..."
                    
                    # Format line
                    line = f"{date_val} | {regime_val:<6d} | {description:<49}"  # Left align description
                    if regime_probs_truncated and i < len(regime_probs_truncated) and regime_probs_truncated[i] and regime_val < len(regime_probs_truncated[i]):
                        line += f" | {regime_probs_truncated[i][regime_val]:.4f}"
                    
                    f.write(line + "\n")
                
                f.write("\n")
                f.write("Note: Only the last 100 data points are shown.\n\n")
            else:
                f.write("No time series data available to display.\n\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Regime report generated: {output_path}")
    return output_path


def generate_comparative_report(regime_groups, ticker_stats, output_path):
    """
    Generate a report comparing regimes across multiple assets
    
    Parameters:
    -----------
    regime_groups : dict
        Dictionary mapping regime indices to lists of tickers in that regime
    ticker_stats : dict
        Dictionary mapping tickers to their statistics
    output_path : str
        Path to save the report file
        
    Returns:
    --------
    output_path : str
        Path to the created report file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("CROSS-ASSET REGIME ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Analysis information
        f.write("ANALYSIS INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of assets analyzed: {len(ticker_stats)}\n")
        f.write(f"Number of regime groups: {len(regime_groups)}\n\n")
        
        # Regime group summaries
        f.write("REGIME GROUP SUMMARIES\n")
        f.write("-" * 80 + "\n")
        
        total_tickers = sum(len(tickers) for tickers in regime_groups.values())
        
        for regime, tickers in sorted(regime_groups.items()):
            percentage = (len(tickers) / total_tickers) * 100
            f.write(f"Regime {regime}\n")
            f.write(f"  Assets: {len(tickers)} ({percentage:.1f}%)\n")
            
            # Calculate average statistics for this regime group
            if len(tickers) > 0:
                # Collect all stats
                mean_returns = []
                volatilities = []
                sharpes = []
                atrs = []
                
                for ticker in tickers:
                    if ticker in ticker_stats and 'current_regime_stats' in ticker_stats[ticker]:
                        stats = ticker_stats[ticker]['current_regime_stats']
                        mean_returns.append(stats.get('mean_return', 0))
                        volatilities.append(stats.get('volatility', 0))
                        sharpes.append(stats.get('sharpe', 0))
                        atrs.append(stats.get('mean_atr', 0))
                
                if mean_returns:
                    # Calculate averages
                    avg_return = np.mean(mean_returns)
                    avg_vol = np.mean(volatilities)
                    avg_sharpe = np.mean(sharpes)
                    avg_atr = np.mean(atrs)
                    
                    f.write(f"  Average statistics for this group:\n")
                    f.write(f"    Mean daily return: {avg_return*100:.4f}%\n")
                    f.write(f"    Annualized return: {avg_return*252*100:.2f}%\n")
                    f.write(f"    Volatility (daily): {avg_vol*100:.4f}%\n")
                    f.write(f"    Volatility (annualized): {avg_vol*np.sqrt(252)*100:.2f}%\n")
                    f.write(f"    Sharpe ratio: {avg_sharpe:.4f}\n")
                    f.write(f"    ATR: {avg_atr*100:.4f}%\n")
            
            # List tickers in this regime
            f.write("  Assets in this regime:\n")
            # Format tickers in columns
            ticker_lines = textwrap.wrap(", ".join(tickers), width=70)
            for line in ticker_lines:
                f.write(f"    {line}\n")
            
            f.write("\n")
        
        # Asset details
        f.write("INDIVIDUAL ASSET DETAILS\n")
        f.write("-" * 80 + "\n")
        
        for ticker, stats in sorted(ticker_stats.items()):
            if 'current_regime' not in stats or 'current_regime_stats' not in stats:
                continue
                
            current_regime = stats['current_regime']
            regime_stats = stats['current_regime_stats']
            
            f.write(f"{ticker}\n")
            f.write(f"  Current regime: {current_regime}\n")
            if 'description' in regime_stats:
                f.write(f"  Description: {regime_stats['description']}\n")
            
            f.write(f"  Mean daily return: {regime_stats.get('mean_return', 0)*100:.4f}%\n")
            f.write(f"  Volatility (daily): {regime_stats.get('volatility', 0)*100:.4f}%\n")
            f.write(f"  Sharpe ratio: {regime_stats.get('sharpe', 0):.4f}\n")
            
            if 'recommendations' in stats:
                recs = stats['recommendations']
                f.write(f"  Trading recommendations:\n")
                for key, value in recs.items():
                    if key != 'transition_risk':  # Handle this separately
                        f.write(f"    {key.replace('_', ' ').title()}: {value}\n")
                
                if 'transition_risk' in recs:
                    risk = recs['transition_risk']
                    f.write(f"    Transition Risk: {risk.get('description', '')} ")
                    f.write(f"({risk.get('probability', 0):.2f})\n")
            
            f.write("\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Comparative report generated: {output_path}")
    return output_path
