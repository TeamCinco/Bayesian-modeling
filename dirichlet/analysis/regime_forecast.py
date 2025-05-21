def forecast_next_regime(hmm_model, current_regime, horizon=5):
    """
    Forecast regime probabilities for the next n days
    
    Parameters:
    -----------
    hmm_model : BayesianHMM
        Fitted HMM model
    current_regime : int
        Current regime index
    horizon : int
        Forecast horizon in days
        
    Returns:
    --------
    forecasts : list of list
        Probabilities for each regime at each forecast step
    """
    forecasts = []
    
    for i in range(1, horizon + 1):
        # Forecast regime probabilities i steps ahead
        probs = hmm_model.forecast_regime_proba(current_regime, steps=i)
        forecasts.append(probs)
    
    return forecasts
