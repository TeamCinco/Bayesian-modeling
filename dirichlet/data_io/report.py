def generate_regime_report(results, output_path, ticker=None, include_time_series=False, file_format='md'):
    """
    Generate a report with regime analysis results
    
    Parameters:
    -----------
    results : dict
        Analysis results from analyze_market_regimes
    output_path : str
        Directory or file path to save the report
    ticker : str, optional
        Ticker symbol
    include_time_series : bool, optional
        Whether to include time series data in the report (can make it very large)
    file_format : str, optional
        Output file format ('md' for Markdown, 'txt' for plain text) - only used if output_path is a directory
    """
    import os
    import numpy as np
    from datetime import datetime
    
    # Check if output_path is a directory or file path
    if os.path.isdir(output_path):
        # It's a directory
        file_extension = '.md' if file_format.lower() == 'md' else '.txt'
        report_filename = f"{ticker}_regime_report{file_extension}" if ticker else f"regime_report{file_extension}"
        report_file = os.path.join(output_path, report_filename)
        output_dir = output_path
    else:
        # It's a file path
        report_file = output_path
        output_dir = os.path.dirname(report_file)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Debug print
    print(f"Creating report at: {report_file}")
    
    # Extract data
    dates = results.get('dates', [])
    regimes = results.get('regimes', [])
    regime_stats = results.get('regime_stats', [])
    
    # Check if we have valid data
    has_regimes = len(regimes) > 0 if isinstance(regimes, (list, tuple)) else (isinstance(regimes, np.ndarray) and regimes.size > 0)
    has_stats = len(regime_stats) > 0
    
    if not has_regimes:
        print("No regime data to generate report.")
        return
    
    # Create report file
    report_file = os.path.join(output_dir, f"{ticker}_regime_report.md" if ticker else "regime_report.md")
    
    with open(report_file, 'w') as f:
        # Write header
        f.write(f"# Market Regime Analysis Report\n\n")
        f.write(f"**Ticker:** {ticker if ticker else 'Unknown'}\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write summary
        if isinstance(regimes, np.ndarray):
            n_regimes = int(regimes.max()) + 1 if regimes.size > 0 else 0
        else:
            n_regimes = max(regimes) + 1 if regimes else 0
            
        f.write(f"## Summary\n\n")
        f.write(f"- Number of regimes detected: {n_regimes}\n")
        f.write(f"- Data points analyzed: {len(dates)}\n")
        
        if has_stats:
            f.write(f"- Analysis period: {regime_stats[0].get('start_date', 'Unknown')} to {regime_stats[-1].get('end_date', 'Unknown')}\n")
        
        f.write("\n")
        
        # Write regime statistics
        if has_stats:
            f.write(f"## Regime Characteristics\n\n")
            
            f.write("| Regime | Count | % of Time | Mean Return | Volatility | Sharpe | Description |\n")
            f.write("|--------|-------|-----------|-------------|------------|--------|--------------|\n")
            
            for stat in sorted(regime_stats, key=lambda x: x['regime']):
                regime_num = stat['regime'] + 1  # 1-indexed for readability
                count = stat.get('count', 0)
                percentage = stat.get('percentage', 0)
                mean_return = stat.get('mean_return', 0) * 100  # Convert to percentage
                volatility = stat.get('volatility', 0) * 100  # Convert to percentage
                sharpe = stat.get('sharpe', 0)
                description = stat.get('description', 'No description')
                
                f.write(f"| Regime {regime_num} | {count} | {percentage:.2f}% | {mean_return:.4f}% | {volatility:.4f}% | {sharpe:.4f} | {description} |\n")
            
            f.write("\n")
        
        # Write transition matrix if available
        hmm_model = results.get('hmm_model', None)
        if hmm_model is not None:
            trans_mat = None
            
            # Try to get transition matrix from various attributes
            if hasattr(hmm_model, 'trans_mat') and hmm_model.trans_mat is not None:
                trans_mat = hmm_model.trans_mat
            elif hasattr(hmm_model, 'transmat_') and hmm_model.transmat_ is not None:
                trans_mat = hmm_model.transmat_
                
            if trans_mat is not None:
                f.write(f"## Regime Transition Matrix\n\n")
                
                f.write("*Probability of transitioning from row regime to column regime*\n\n")
                
                # Write table header
                f.write("| From/To |")
                for j in range(n_regimes):
                    f.write(f" Regime {j+1} |")
                f.write("\n")
                
                # Write table separator
                f.write("|---------|")
                for j in range(n_regimes):
                    f.write("---------|")
                f.write("\n")
                
                # Write transition probabilities
                for i in range(n_regimes):
                    f.write(f"| Regime {i+1} |")
                    for j in range(n_regimes):
                        if i < len(trans_mat) and j < len(trans_mat[i]):
                            f.write(f" {trans_mat[i][j]:.4f} |")
                        else:
                            f.write(" N/A |")
                    f.write("\n")
                
                f.write("\n")
        
        # Write current regime analysis if available
        if 'current_regime' in results:
            current_regime = results['current_regime']
            
            f.write(f"## Current Market Regime\n\n")
            f.write(f"The current market regime is: **Regime {current_regime + 1}**\n\n")
            
            # Find stats for current regime
            current_stats = None
            for stat in regime_stats:
                if stat['regime'] == current_regime:
                    current_stats = stat
                    break
            
            if current_stats:
                f.write("### Characteristics of Current Regime\n\n")
                f.write(f"- Description: {current_stats.get('description', 'No description')}\n")
                f.write(f"- Mean Daily Return: {current_stats.get('mean_return', 0) * 100:.4f}%\n")
                f.write(f"- Volatility: {current_stats.get('volatility', 0) * 100:.4f}%\n")
                f.write(f"- Sharpe Ratio: {current_stats.get('sharpe', 0):.4f}\n")
        
        # Write recommendations if available
        if 'recommendations' in results:
            recommendations = results['recommendations']
            
            f.write(f"## Strategy Recommendations\n\n")
            
            f.write("### Primary Strategy\n\n")
            f.write(f"{recommendations.get('primary_strategy', 'No recommendation')}\n\n")
            
            f.write("### Alternative Strategy\n\n")
            f.write(f"{recommendations.get('alternative_strategy', 'No recommendation')}\n\n")
            
            f.write("### Position Sizing\n\n")
            f.write(f"{recommendations.get('position_sizing', 'No recommendation')}\n\n")
            
            if 'confidence' in recommendations:
                f.write(f"*Confidence: {recommendations['confidence']}*\n\n")
        
        # Write stability analysis if available
        if 'forecasts' in results and 'stability' in results['forecasts']:
            stability = results['forecasts']['stability']
            
            f.write(f"## Regime Stability Analysis\n\n")
            f.write(f"- Stability: **{stability.get('stability_description', 'Unknown')}**\n")
            f.write(f"- Transition Risk: {stability.get('transition_risk', 'Unknown')}\n")
            f.write(f"- Expected Remaining Duration: {stability.get('expected_remaining_duration', 'Unknown')} days\n")
            f.write(f"- Most Likely Next Regime: Regime {stability.get('most_likely_next', 0) + 1}\n\n")
            
            f.write("*Note: Stability analysis is based on historical patterns and should be used as guidance only.*\n\n")
        
        # Include time series data if requested
        if include_time_series and len(dates) > 0 and has_regimes:
            f.write(f"## Time Series Data\n\n")
            
            # Determine how many data points to include (limit to avoid huge reports)
            max_rows = 100  # Adjust as needed
            
            # Get regime probabilities if available
            regime_probs = results.get('regime_probs', [])
            has_probs = len(regime_probs) > 0
            
            # Create table header
            f.write("| Date | Regime |")
            if 'ohlcv' in results and 'close' in results['ohlcv']:
                f.write(" Close |")
            if 'features' in results and 'log_returns' in results['features']:
                f.write(" Return |")
            if has_probs:
                for i in range(n_regimes):
                    f.write(f" Prob R{i+1} |")
            f.write("\n")
            
            # Create table separator
            f.write("|------|--------|")
            if 'ohlcv' in results and 'close' in results['ohlcv']:
                f.write("-------|")
            if 'features' in results and 'log_returns' in results['features']:
                f.write("--------|")
            if has_probs:
                for i in range(n_regimes):
                    f.write("---------|")
            f.write("\n")
            
            # Determine sampling if too many rows
            stride = max(1, len(dates) // max_rows)
            
            # Write rows
            for i in range(0, len(dates), stride):
                # Skip first row for returns
                if i == 0 and 'features' in results and 'log_returns' in results['features']:
                    continue
                    
                if i >= len(dates) or (i > 0 and i-1 >= len(regimes)):
                    continue
                    
                date = dates[i]
                regime = regimes[i-1] + 1 if i > 0 else 'N/A'  # 1-indexed, returns shifted by 1
                
                f.write(f"| {date} | {regime} |")
                
                if 'ohlcv' in results and 'close' in results['ohlcv'] and i < len(results['ohlcv']['close']):
                    f.write(f" {results['ohlcv']['close'][i]:.2f} |")
                    
                if 'features' in results and 'log_returns' in results['features'] and i > 0 and i-1 < len(results['features']['log_returns']):
                    ret = results['features']['log_returns'][i-1] * 100  # Convert to percentage
                    f.write(f" {ret:.4f}% |")
                    
                if has_probs and i > 0 and i-1 < len(regime_probs):
                    probs = regime_probs[i-1]
                    for j in range(min(n_regimes, len(probs))):
                        f.write(f" {probs[j]:.4f} |")
                    for j in range(len(probs), n_regimes):
                        f.write(" N/A |")
                        
                f.write("\n")
            
            if stride > 1:
                f.write("\n*Note: Time series data has been sampled to limit report size.*\n\n")
    
    print(f"Report generated: {report_file}")

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
