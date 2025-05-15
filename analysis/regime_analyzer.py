import math

def analyze_regime_statistics(regimes, features, ohlcv, dates):
    """
    Calculate statistics for each regime
    
    Parameters:
    -----------
    regimes : list
        List of regime assignments for each time point
    features : dict
        Dictionary of extracted features
    ohlcv : dict
        Dictionary with OHLCV data
    dates : list
        List of dates
        
    Returns:
    --------
    stats : list
        List of dictionaries with statistics for each regime
    """
    n_regimes = max(regimes) + 1
    stats = []
    
    # Extract needed features
    returns = features['log_returns']
    volatility = features['volatility']
    atr = features['atr']
    
    for regime in range(n_regimes):
        # Find indices for this regime
        indices = [i for i in range(len(regimes)) if regimes[i] == regime]
        
        if not indices:
            continue
        
        # Collect regime data
        regime_returns = [returns[i] for i in indices]
        regime_volatility = [volatility[i] for i in indices]
        regime_atr = [atr[i] for i in indices]
        
        # Calculate statistics
        count = len(indices)
        percentage = (count / len(regimes)) * 100
        
        # Return statistics
        mean_return = sum(regime_returns) / count if count > 0 else 0
        regime_vol = math.sqrt(sum(r**2 for r in regime_returns) / count) if count > 0 else 0
        
        # Sharpe ratio (annualized)
        sharpe = (mean_return * 252) / (regime_vol * math.sqrt(252)) if regime_vol > 0 else 0
        
        # ATR statistics
        mean_atr = sum(regime_atr) / count if count > 0 else 0
        
        # Date ranges
        start_date = dates[indices[0]] if indices else None
        end_date = dates[indices[-1]] if indices else None
        
        # Create description based on statistics
        description = regime_description(mean_return, regime_vol, sharpe, mean_atr)
        
        # Add to statistics
        stats.append({
            'regime': regime,
            'count': count,
            'percentage': percentage,
            'mean_return': mean_return,
            'volatility': regime_vol,
            'sharpe': sharpe,
            'mean_atr': mean_atr,
            'start_date': start_date,
            'end_date': end_date,
            'description': description
        })
    
    return stats


def regime_description(mean_return, volatility, sharpe, atr):
    """
    Generate descriptive label for a market regime
    
    Parameters:
    -----------
    mean_return : float
        Mean log return during the regime
    volatility : float
        Standard deviation of returns during the regime
    sharpe : float
        Sharpe ratio during the regime
    atr : float
        Average True Range during the regime
        
    Returns:
    --------
    description : str
        Human-readable description of the regime
    """
    description = []
    
    # Volatility description
    if volatility < 0.005:
        description.append("Low Volatility")
    elif volatility < 0.015:
        description.append("Medium Volatility")
    else:
        description.append("High Volatility")
    
    # Return description
    if mean_return < -0.001:
        description.append("Bearish")
    elif mean_return < 0.001:
        description.append("Sideways")
    else:
        description.append("Bullish")
    
    # Sharpe description
    if abs(sharpe) < 0.5:
        description.append("Inefficient")
    elif sharpe > 1.0:
        description.append("Efficient")
    
    # ATR description
    if atr < 0.01:
        description.append("Low Range")
    elif atr > 0.02:
        description.append("Wide Range")
    
    return ", ".join(description)


def sort_regimes_by_volatility(regimes, regime_stats):
    """
    Sort regimes by volatility for better interpretation
    
    Parameters:
    -----------
    regimes : list
        List of regime assignments
    regime_stats : list
        List of regime statistics
        
    Returns:
    --------
    sorted_regimes : list
        New regime assignments sorted by volatility
    regime_mapping : dict
        Mapping from old regime indices to new indices
    """
    # Sort regime stats by volatility
    sorted_stats = sorted(regime_stats, key=lambda x: x['volatility'])
    
    # Create mapping from old regime to new regime
    regime_mapping = {stat['regime']: i for i, stat in enumerate(sorted_stats)}
    
    # Map regimes to new indices
    sorted_regimes = [regime_mapping[r] for r in regimes]
    
    return sorted_regimes, regime_mapping


def analyze_market_regimes(dates, ohlcv, features, hmm, n_regimes=3):
    """
    Analyze market regimes using Bayesian HMM
    
    Parameters:
    -----------
    dates : list
        List of dates
    ohlcv : dict
        Dictionary with OHLCV data
    features : dict
        Dictionary of extracted features
    hmm : BayesianHMM
        Fitted Bayesian HMM model
    n_regimes : int
        Number of regimes to identify
        
    Returns:
    --------
    results : dict
        Analysis results including regime assignments and statistics
    """
    # Get regime assignments
    combined_feature = features['combined']
    regimes = hmm.predict(combined_feature)
    
    # Get regime probabilities
    regime_probs = hmm.predict_proba(combined_feature)
    
    # Analyze regime characteristics
    regime_stats = analyze_regime_statistics(regimes, features, ohlcv, dates)
    
    # Sort regimes by volatility for better interpretation
    sorted_regimes, regime_mapping = sort_regimes_by_volatility(regimes, regime_stats)
    
    # Map regime probabilities to sorted regimes
    sorted_regime_probs = []
    for t in range(len(regime_probs)):
        probs = [0.0] * n_regimes
        for old_regime, new_regime in regime_mapping.items():
            probs[new_regime] = regime_probs[t][old_regime]
        sorted_regime_probs.append(probs)
    
    # Create results dictionary
    results = {
        'dates': dates,
        'ohlcv': ohlcv,
        'features': features,
        'regimes': sorted_regimes,
        'regime_probs': sorted_regime_probs,
        'regime_stats': regime_stats,
        'hmm_model': hmm,
        'regime_mapping': regime_mapping
    }
    
    return results


def analyze_cross_asset_regimes(summary_file_path):
    """
    Analyze regime correlations across multiple assets
    
    Parameters:
    -----------
    summary_file_path : str
        Path to the summary file containing regime assignments for multiple assets
        
    Returns:
    --------
    regime_groups : dict
        Dictionary mapping regime indices to lists of tickers in that regime
    """
    import csv
    import os
    
    if not os.path.exists(summary_file_path):
        print("No summary file found. Run the main analysis first.")
        return None
    
    # Load summary data
    tickers = []
    current_regimes = {}
    
    with open(summary_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row['Ticker']
            try:
                regime = int(row['Current_Regime'])
                tickers.append(ticker)
                current_regimes[ticker] = regime
            except:
                # Skip rows with errors
                continue
    
    # Group tickers by regime
    regime_groups = {}
    for ticker, regime in current_regimes.items():
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(ticker)
    
    # Print cross-asset regime analysis
    print("\n=== CROSS-ASSET REGIME ANALYSIS ===")
    print(f"Total assets analyzed: {len(tickers)}")
    
    for regime, ticker_list in sorted(regime_groups.items()):
        percentage = (len(ticker_list) / len(tickers)) * 100
        print(f"Regime {regime}: {len(ticker_list)} assets ({percentage:.1f}%)")
        print(f"  Assets: {', '.join(ticker_list[:10])}" + 
              (f" and {len(ticker_list)-10} more..." if len(ticker_list) > 10 else ""))
    
    return regime_groups
