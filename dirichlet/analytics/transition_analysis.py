"""
Transition analysis for market regimes.

This module provides functions to analyze transitions between market
regimes, calculate regime durations, and predict regime stability.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter


def calculate_transition_probabilities(regimes):
    """
    Calculate transition probabilities between regimes
    
    Parameters:
    -----------
    regimes : list
        List of regime assignments
        
    Returns:
    --------
    transition_matrix : numpy.ndarray
        Matrix of transition probabilities
    transition_counts : numpy.ndarray
        Matrix of transition counts
    """
    n_regimes = max(regimes) + 1
    
    # Initialize transition counts
    transition_counts = np.zeros((n_regimes, n_regimes))
    
    # Count transitions
    for i in range(len(regimes) - 1):
        from_regime = regimes[i]
        to_regime = regimes[i + 1]
        transition_counts[from_regime, to_regime] += 1
    
    # Calculate probabilities
    transition_matrix = np.zeros_like(transition_counts, dtype=float)
    
    for i in range(n_regimes):
        row_sum = transition_counts[i].sum()
        if row_sum > 0:
            transition_matrix[i] = transition_counts[i] / row_sum
    
    return transition_matrix, transition_counts


def analyze_regime_duration(regimes, dates=None):
    """
    Analyze the duration of each regime
    
    Parameters:
    -----------
    regimes : list
        List of regime assignments
    dates : list, optional
        List of dates corresponding to regimes
        
    Returns:
    --------
    duration_stats : dict
        Dictionary containing duration statistics for each regime
    """
    n_regimes = max(regimes) + 1
    duration_stats = {}
    
    # If dates are not provided, create an index
    if dates is None:
        dates = list(range(len(regimes)))
    
    # Process each regime
    for r in range(n_regimes):
        # Find all occurrences of this regime
        occurrences = []
        current_start = None
        current_duration = 0
        
        for i, regime in enumerate(regimes):
            if regime == r:
                if current_start is None:
                    current_start = i
                current_duration += 1
            else:
                if current_start is not None:
                    occurrences.append({
                        'start_idx': current_start,
                        'end_idx': i - 1,
                        'start_date': dates[current_start],
                        'end_date': dates[i - 1],
                        'duration': current_duration
                    })
                    current_start = None
                    current_duration = 0
        
        # Add the last occurrence if it extends to the end
        if current_start is not None:
            occurrences.append({
                'start_idx': current_start,
                'end_idx': len(regimes) - 1,
                'start_date': dates[current_start],
                'end_date': dates[-1],
                'duration': current_duration
            })
        
        # Calculate statistics
        if occurrences:
            durations = [occ['duration'] for occ in occurrences]
            
            duration_stats[r] = {
                'occurrences': occurrences,
                'mean_duration': sum(durations) / len(durations),
                'median_duration': np.median(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations),
                'count': len(occurrences)
            }
        else:
            duration_stats[r] = {
                'occurrences': [],
                'mean_duration': 0,
                'median_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'total_duration': 0,
                'count': 0
            }
    
    return duration_stats


def calculate_transition_likelihood(current_regime, duration, transition_matrix, duration_stats):
    """
    Calculate the likelihood of transitioning from the current regime
    
    Parameters:
    -----------
    current_regime : int
        Current regime index
    duration : int
        Number of days spent in the current regime
    transition_matrix : numpy.ndarray
        Matrix of transition probabilities
    duration_stats : dict
        Dictionary containing duration statistics for each regime
        
    Returns:
    --------
    likelihood : float
        Likelihood of transitioning to another regime
    most_likely_next : int
        Most likely next regime
    """
    # Basic transition probability from matrix
    base_probs = transition_matrix[current_regime].copy()
    
    # Adjust probability based on duration
    if current_regime in duration_stats:
        stats = duration_stats[current_regime]
        
        if stats['mean_duration'] > 0:
            # If we've been in this regime for longer than average,
            # increase the probability of transitioning
            duration_factor = duration / stats['mean_duration']
            
            # Adjust the staying probability (probability of transitioning to same regime)
            stay_prob = base_probs[current_regime]
            
            if duration_factor > 1:
                # Reduce the probability of staying in the same regime
                adjustment = min(stay_prob * (duration_factor - 1) / 2, stay_prob * 0.8)
                base_probs[current_regime] = stay_prob - adjustment
                
                # Distribute the adjustment to other regimes proportionally
                non_stay_total = 1 - stay_prob
                if non_stay_total > 0:
                    for i in range(len(base_probs)):
                        if i != current_regime:
                            base_probs[i] += adjustment * (base_probs[i] / non_stay_total)
    
    # Normalize probabilities
    total_prob = base_probs.sum()
    if total_prob > 0:
        base_probs = base_probs / total_prob
    
    # Get likelihood of transitioning to any other regime
    likelihood = 1 - base_probs[current_regime]
    
    # Get most likely next regime
    if np.sum(base_probs) > 0:
        most_likely_next = np.argmax(base_probs)
    else:
        most_likely_next = current_regime
    
    return likelihood, most_likely_next


def predict_regime_stability(current_regime, duration, regimes, dates=None):
    """
    Predict the stability of the current regime
    
    Parameters:
    -----------
    current_regime : int
        Current regime index
    duration : int
        Number of days spent in the current regime
    regimes : list
        Historical regime assignments
    dates : list, optional
        List of dates corresponding to regimes
        
    Returns:
    --------
    stability : dict
        Dictionary containing stability metrics
    """
    # Calculate transition probabilities
    trans_matrix, _ = calculate_transition_probabilities(regimes)
    
    # Analyze durations
    duration_stats = analyze_regime_duration(regimes, dates)
    
    # Calculate transition likelihood
    likelihood, next_regime = calculate_transition_likelihood(
        current_regime, duration, trans_matrix, duration_stats
    )
    
    # Compare current duration to historical durations
    regime_stats = duration_stats.get(current_regime, {})
    mean_duration = regime_stats.get('mean_duration', 0)
    max_duration = regime_stats.get('max_duration', 0)
    
    if mean_duration > 0:
        # Calculate percentile of current duration
        durations = [occ['duration'] for occ in regime_stats.get('occurrences', [])]
        if durations:
            percentile = sum(1 for d in durations if d <= duration) / len(durations)
        else:
            percentile = 0
    else:
        percentile = 0
    
    # Calculate expected remaining duration based on historical data
    if mean_duration > duration:
        expected_remaining = mean_duration - duration
    else:
        expected_remaining = 1  # Minimum
    
    # Create stability description
    if duration < mean_duration * 0.5:
        stability_desc = "Very Stable"
        risk_level = "Low"
    elif duration < mean_duration:
        stability_desc = "Stable"
        risk_level = "Low to Medium"
    elif duration < mean_duration * 1.5:
        stability_desc = "Moderately Stable"
        risk_level = "Medium"
    elif duration < max_duration:
        stability_desc = "Less Stable"
        risk_level = "Medium to High"
    else:
        stability_desc = "Unstable"
        risk_level = "High"
    
    # Create result dictionary
    stability = {
        'current_regime': current_regime,
        'current_duration': duration,
        'mean_duration': mean_duration,
        'max_duration': max_duration,
        'duration_percentile': percentile,
        'transition_likelihood': likelihood,
        'most_likely_next': next_regime,
        'expected_remaining_duration': max(1, int(expected_remaining)),
        'stability_description': stability_desc,
        'transition_risk': risk_level
    }
    
    return stability


def predict_regime_transitions(current_regime, hmm, horizon=5, n_simulations=1000):
    """
    Predict future regime transitions using Monte Carlo simulation
    
    Parameters:
    -----------
    current_regime : int
        Current regime index
    hmm : BayesianHMM
        Fitted HMM model
    horizon : int, optional
        Number of periods to forecast
    n_simulations : int, optional
        Number of simulations to run
        
    Returns:
    --------
    forecast : dict
        Dictionary containing forecast results
    """
    # Get transition matrix
    transition_matrix = hmm.transmat_
    
    # Initialize counters for each period
    regime_counts = []
    for _ in range(horizon):
        regime_counts.append(Counter())
    
    # Run simulations
    for _ in range(n_simulations):
        regime = current_regime
        
        for t in range(horizon):
            # Random transition based on probabilities
            regime = np.random.choice(len(transition_matrix), p=transition_matrix[regime])
            regime_counts[t][regime] += 1
    
    # Calculate probabilities for each period
    regime_probs = []
    
    for t in range(horizon):
        probs = {}
        for regime, count in regime_counts[t].items():
            probs[regime] = count / n_simulations
        regime_probs.append(probs)
    
    # Find most likely sequence
    most_likely_path = [current_regime]
    regime = current_regime
    
    for t in range(horizon):
        # Get most likely transition from current regime
        next_regime = np.argmax(transition_matrix[regime])
        most_likely_path.append(next_regime)
        regime = next_regime
    
    # Prepare forecast results
    forecast = {
        'current_regime': current_regime,
        'horizon': horizon,
        'regime_probabilities': regime_probs,
        'most_likely_path': most_likely_path[1:],  # Remove the current regime
        'transition_matrix': transition_matrix.tolist()
    }
    
    return forecast
