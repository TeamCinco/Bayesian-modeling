from .algorithms import viterbi_algorithm, forward_backward_algorithm

def predict_states(observations, n_regimes, init_probs, trans_mat, mu, sigma):
    """
    Predict most likely regime sequence for observations
    
    Returns:
    --------
    states : list
        Most likely hidden state sequence
    """
    return viterbi_algorithm(observations, n_regimes, init_probs, trans_mat, mu, sigma)

def predict_probabilities(observations, n_regimes, init_probs, trans_mat, mu, sigma):
    """
    Predict regime probabilities for observations
    
    Returns:
    --------
    probabilities : list of lists
        Probability of each regime at each time step
    """
    return forward_backward_algorithm(observations, n_regimes, init_probs, trans_mat, mu, sigma)

def forecast_regime_proba(current_regime, steps, n_regimes, trans_mat):
    """
    Forecast regime probabilities n steps ahead
    
    Returns:
    --------
    probabilities : list
        Probability of each regime after n steps
    """
    probs = [0.0] * n_regimes
    probs[current_regime] = 1.0
    
    # Apply transition matrix 'steps' times
    for _ in range(steps):
        new_probs = [0.0] * n_regimes
        for i in range(n_regimes):
            for j in range(n_regimes):
                new_probs[j] += probs[i] * trans_mat[i][j]
        probs = new_probs
    
    return probs
