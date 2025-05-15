"""
Extended statistical analysis for market regimes.

This module provides functions to perform detailed statistical analysis
of market regimes, including risk-adjusted metrics, correlations,
and higher moments of return distributions.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats


def calculate_extended_regime_statistics(regimes, features, ohlcv, dates, annualization=252):
    """
    Calculate extended statistics for each regime
    
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
    annualization : int, optional
        Annualization factor (default: 252 for daily data)
        
    Returns:
    --------
    extended_stats : list
        List of dictionaries with extended statistics for each regime
    """
    n_regimes = max(regimes) + 1
    extended_stats = []
    
    # Extract needed features
    returns = features['log_returns']
    volatility = features.get('volatility', [0] * len(returns))
    atr = features.get('atr', [0] * len(returns))
    
    for regime in range(n_regimes):
        # Find indices for this regime
        indices = [i for i in range(len(regimes)) if regimes[i] == regime]
        
        if not indices:
            continue
        
        # Collect regime data
        regime_returns = [returns[i] for i in indices]
        regime_volatility = [volatility[i] for i in indices if i < len(volatility)]
        regime_atr = [atr[i] for i in indices if i < len(atr)]
        regime_dates = [dates[i] for i in indices]
        regime_prices = [ohlcv['close'][i] for i in indices if i < len(ohlcv['close'])]
        
        # Basic statistics
        count = len(indices)
        percentage = (count / len(regimes)) * 100
        
        # Return statistics
        mean_return = sum(regime_returns) / count if count > 0 else 0
        regime_vol = np.std(regime_returns) if count > 1 else 0
        
        # Higher moments
        skewness = stats.skew(regime_returns) if count > 2 else 0
        kurtosis = stats.kurtosis(regime_returns) if count > 2 else 0
        
        # Date ranges and durations
        start_date = regime_dates[0] if regime_dates else None
        end_date = regime_dates[-1] if regime_dates else None
        
        # Calculate regime durations (continuous periods)
        durations = []
        current_duration = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 1
                
        # Add the last duration
        if current_duration > 0:
            durations.append(current_duration)
        
        # Calculate duration statistics
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        
        # Risk metrics
        if count > 1:
            # Downside deviation (semi-deviation)
            neg_returns = [r for r in regime_returns if r < 0]
            downside_dev = np.std(neg_returns) if neg_returns else 0
            
            # Value at Risk (95%)
            var_95 = np.percentile(regime_returns, 5)
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = np.mean([r for r in regime_returns if r <= var_95]) if var_95 < 0 else 0
            
            # Maximum drawdown
            if regime_prices:
                peak = regime_prices[0]
                max_drawdown = 0
                
                for price in regime_prices:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0
        else:
            downside_dev = 0
            var_95 = 0
            cvar_95 = 0
            max_drawdown = 0
        
        # Risk-adjusted performance metrics
        if regime_vol > 0:
            # Sharpe ratio (annualized)
            sharpe = (mean_return * annualization) / (regime_vol * math.sqrt(annualization))
            
            # Sortino ratio (using downside deviation)
            sortino = (mean_return * annualization) / (downside_dev * math.sqrt(annualization)) if downside_dev > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            calmar = (mean_return * annualization) / max_drawdown if max_drawdown > 0 else 0
        else:
            sharpe = 0
            sortino = 0
            calmar = 0
        
        # Add to statistics
        extended_stats.append({
            'regime': regime,
            'count': count,
            'percentage': percentage,
            
            # Basic return statistics
            'mean_return': mean_return,
            'median_return': np.median(regime_returns) if count > 0 else 0,
            'min_return': min(regime_returns) if count > 0 else 0,
            'max_return': max(regime_returns) if count > 0 else 0,
            'volatility': regime_vol,
            
            # Higher moments
            'skewness': skewness,
            'kurtosis': kurtosis,
            
            # Risk metrics
            'downside_deviation': downside_dev,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            
            # Risk-adjusted metrics
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            
            # Duration information
            'avg_duration': avg_duration,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'durations': durations,
            
            # Date information
            'start_date': start_date,
            'end_date': end_date,
            
            # ATR information
            'mean_atr': sum(regime_atr) / count if count > 0 and regime_atr else 0,
            'max_atr': max(regime_atr) if regime_atr else 0,
            'min_atr': min(regime_atr) if regime_atr else 0,
        })
    
    return extended_stats


def calculate_regime_correlations(regimes, other_series, labels=None):
    """
    Calculate correlations between regime indicators and other time series
    
    Parameters:
    -----------
    regimes : list
        List of regime assignments for each time point
    other_series : list of lists
        List of other time series to correlate with regimes
    labels : list, optional
        Labels for the other time series
        
    Returns:
    --------
    correlations : pd.DataFrame
        DataFrame with correlation values
    """
    n_regimes = max(regimes) + 1
    n_series = len(other_series)
    
    # Create regime indicator series (one-hot encoding)
    regime_indicators = []
    for r in range(n_regimes):
        indicator = [1 if regime == r else 0 for regime in regimes]
        regime_indicators.append(indicator)
    
    # Create a DataFrame
    data = {}
    
    # Add regime indicators
    for r in range(n_regimes):
        data[f'Regime_{r}'] = regime_indicators[r]
    
    # Add other series
    if labels is None:
        labels = [f'Series_{i}' for i in range(n_series)]
    
    for i, series in enumerate(other_series):
        if len(series) != len(regimes):
            # Trim or pad to match length
            if len(series) > len(regimes):
                series = series[:len(regimes)]
            else:
                series = series + [0] * (len(regimes) - len(series))
                
        data[labels[i]] = series
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    # Extract only correlations between regimes and other series
    regime_cols = [col for col in correlation_matrix.columns if col.startswith('Regime_')]
    other_cols = [col for col in correlation_matrix.columns if not col.startswith('Regime_')]
    
    correlations = correlation_matrix.loc[regime_cols, other_cols]
    
    return correlations


def analyze_risk_adjusted_metrics(extended_stats, benchmark_return=0.0):
    """
    Analyze risk-adjusted performance metrics for each regime
    
    Parameters:
    -----------
    extended_stats : list
        List of dictionaries with extended statistics for each regime
    benchmark_return : float, optional
        Benchmark return for information ratio calculation
        
    Returns:
    --------
    risk_metrics : list
        List of dictionaries with risk-adjusted metrics for each regime
    """
    risk_metrics = []
    
    for stat in extended_stats:
        regime = stat['regime']
        mean_return = stat['mean_return']
        volatility = stat['volatility']
        
        # Calculate relative performance
        excess_return = mean_return - benchmark_return
        
        # Information Ratio
        information_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Treynor Ratio (assuming beta = volatility/benchmark_volatility)
        # Since we don't have a true beta, we'll use volatility as a proxy
        treynor_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Calculate Gain-to-Pain metrics
        if 'downside_deviation' in stat and stat['downside_deviation'] > 0:
            gain_to_pain = mean_return / stat['downside_deviation']
        else:
            gain_to_pain = 0
        
        # Create risk metrics dictionary
        metrics = {
            'regime': regime,
            'excess_return': excess_return,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'gain_to_pain': gain_to_pain,
            'sharpe_ratio': stat.get('sharpe', 0),
            'sortino_ratio': stat.get('sortino', 0),
            'calmar_ratio': stat.get('calmar', 0),
            'var_95': stat.get('var_95', 0),
            'cvar_95': stat.get('cvar_95', 0),
            'max_drawdown': stat.get('max_drawdown', 0)
        }
        
        risk_metrics.append(metrics)
    
    return risk_metrics
