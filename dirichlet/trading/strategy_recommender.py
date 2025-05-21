def recommend_option_strategy(regime_stats, current_regime, forecasts):
    """
    Recommend option strategies based on current regime and forecasts
    
    Parameters:
    -----------
    regime_stats : list
        Statistics for each regime
    current_regime : int
        Current regime index
    forecasts : list of list
        Probability forecasts for future regimes
        
    Returns:
    --------
    recommendations : dict
        Strategy recommendations
    """
    # Get current regime characteristics
    current_stats = regime_stats[current_regime]
    
    # Initialize recommendations
    recommendations = {
        'current_regime': current_stats['description'],
        'regime_characteristics': current_stats,
        'primary_strategy': None,
        'alternative_strategy': None,
        'position_sizing': None,
        'wing_width': None,
        'expiration': None,
        'confidence': None,
        'transition_risk': None
    }
    
    # Determine primary strategy based on regime
    volatility = current_stats['volatility']
    mean_return = current_stats['mean_return']
    sharpe = current_stats['sharpe']
    
    # Analyze transition risk (probability of changing regimes)
    next_day_forecast = forecasts[0]
    transition_prob = 1.0 - next_day_forecast[current_regime]
    
    recommendations['transition_risk'] = {
        'probability': transition_prob,
        'description': "High" if transition_prob > 0.3 else "Medium" if transition_prob > 0.1 else "Low"
    }
    
    # Primary strategy recommendation
    if volatility < 0.01:  # Low volatility
        if mean_return > 0:  # Bullish
            recommendations['primary_strategy'] = "Iron Condor (with bullish skew)"
            recommendations['alternative_strategy'] = "Put Credit Spread"
            recommendations['wing_width'] = "Narrow (0.5-1 standard deviation)"
            recommendations['position_sizing'] = "Larger (70-80% of max allocation)"
        elif mean_return < -0.0005:  # Bearish
            recommendations['primary_strategy'] = "Iron Condor (with bearish skew)"
            recommendations['alternative_strategy'] = "Call Credit Spread"
            recommendations['wing_width'] = "Narrow (0.5-1 standard deviation)"
            recommendations['position_sizing'] = "Larger (70-80% of max allocation)"
        else:  # Sideways
            recommendations['primary_strategy'] = "Iron Condor (symmetric)"
            recommendations['alternative_strategy'] = "Short Strangle (if approved for naked options)"
            recommendations['wing_width'] = "Very Narrow (0.5 standard deviation)"
            recommendations['position_sizing'] = "Larger (80-90% of max allocation)"
    
    elif volatility < 0.02:  # Medium volatility
        if mean_return > 0.001:  # Strongly bullish
            recommendations['primary_strategy'] = "Put Credit Spread"
            recommendations['alternative_strategy'] = "Call Debit Spread"
            recommendations['wing_width'] = "Medium (1-1.5 standard deviations)"
            recommendations['position_sizing'] = "Moderate (50-70% of max allocation)"
        elif mean_return < -0.001:  # Strongly bearish
            recommendations['primary_strategy'] = "Call Credit Spread"
            recommendations['alternative_strategy'] = "Put Debit Spread"
            recommendations['wing_width'] = "Medium (1-1.5 standard deviations)"
            recommendations['position_sizing'] = "Moderate (50-70% of max allocation)"
        else:  # Slightly directional or sideways
            recommendations['primary_strategy'] = "Iron Condor"
            recommendations['alternative_strategy'] = "Iron Butterfly"
            recommendations['wing_width'] = "Medium (1 standard deviation)"
            recommendations['position_sizing'] = "Moderate (50-70% of max allocation)"
    
    else:  # High volatility
        if transition_prob > 0.3:  # High transition risk
            recommendations['primary_strategy'] = "Avoid new positions"
            recommendations['alternative_strategy'] = "Very small Long Call or Put (directional bias)"
            recommendations['wing_width'] = "N/A"
            recommendations['position_sizing'] = "Minimal (0-30% of max allocation)"
        elif mean_return > 0.002:  # Strongly bullish in high vol
            recommendations['primary_strategy'] = "Call Debit Spread"
            recommendations['alternative_strategy'] = "Long Call"
            recommendations['wing_width'] = "Wide (1.5-2 standard deviations)"
            recommendations['position_sizing'] = "Small (30-50% of max allocation)"
        elif mean_return < -0.002:  # Strongly bearish in high vol
            recommendations['primary_strategy'] = "Put Debit Spread"
            recommendations['alternative_strategy'] = "Long Put"
            recommendations['wing_width'] = "Wide (1.5-2 standard deviations)"
            recommendations['position_sizing'] = "Small (30-50% of max allocation)"
        else:  # High vol but unclear direction
            recommendations['primary_strategy'] = "Calendar Spread"
            recommendations['alternative_strategy'] = "Diagonal Spread"
            recommendations['wing_width'] = "Wide (1.5-2 standard deviations)"
            recommendations['position_sizing'] = "Small (30-50% of max allocation)"
    
    # Determine appropriate expiration based on forecast stability
    regime_stability = 1.0 - transition_prob
    if regime_stability > 0.7:
        recommendations['expiration'] = "30-45 days"
    elif regime_stability > 0.5:
        recommendations['expiration'] = "21-30 days"
    else:
        recommendations['expiration'] = "7-14 days"
    
    # Determine confidence level based on model performance and forecast certainty
    if max(next_day_forecast) > 0.7 and transition_prob < 0.2:
        recommendations['confidence'] = "High"
    elif max(next_day_forecast) > 0.5:
        recommendations['confidence'] = "Medium"
    else:
        recommendations['confidence'] = "Low"
    
    return recommendations
