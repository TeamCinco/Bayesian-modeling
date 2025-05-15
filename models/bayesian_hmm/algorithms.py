from .math_utils import gaussian_pdf

def forward_algorithm(observations, n_regimes, init_probs, trans_mat, mu, sigma):
    """Forward algorithm for computing alpha variables"""
    T = len(observations)
    alpha = [[0.0 for _ in range(n_regimes)] for _ in range(T)]
    
    # Initialize
    for j in range(n_regimes):
        alpha[0][j] = init_probs[j] * gaussian_pdf(observations[0], mu[j], sigma[j])
    
    # Normalize to prevent underflow
    alpha_sum = sum(alpha[0])
    if alpha_sum > 0:
        for j in range(n_regimes):
            alpha[0][j] /= alpha_sum
    
    # Recursion
    for t in range(1, T):
        for j in range(n_regimes):
            alpha[t][j] = sum(alpha[t-1][i] * trans_mat[i][j] for i in range(n_regimes))
            alpha[t][j] *= gaussian_pdf(observations[t], mu[j], sigma[j])
        
        # Normalize
        alpha_sum = sum(alpha[t])
        if alpha_sum > 0:
            for j in range(n_regimes):
                alpha[t][j] /= alpha_sum
    
    return alpha

def backward_algorithm(observations, n_regimes, trans_mat, mu, sigma):
    """Backward algorithm for computing beta variables"""
    T = len(observations)
    beta = [[0.0 for _ in range(n_regimes)] for _ in range(T)]
    
    # Initialize
    for j in range(n_regimes):
        beta[T-1][j] = 1.0
    
    # Recursion
    for t in range(T-2, -1, -1):
        for i in range(n_regimes):
            beta[t][i] = 0.0
            for j in range(n_regimes):
                beta[t][i] += trans_mat[i][j] * gaussian_pdf(observations[t+1], mu[j], sigma[j]) * beta[t+1][j]
        
        # Normalize
        beta_sum = sum(beta[t])
        if beta_sum > 0:
            for i in range(n_regimes):
                beta[t][i] /= beta_sum
    
    return beta

def forward_backward_algorithm(observations, n_regimes, init_probs, trans_mat, mu, sigma):
    """
    Run forward-backward algorithm on observation sequence
    
    Returns:
    --------
    gamma : list of lists
        Posterior probabilities P(S_t=j|observations) for each time step and regime
    """
    T = len(observations)
    
    # Forward pass
    alpha = forward_algorithm(observations, n_regimes, init_probs, trans_mat, mu, sigma)
    
    # Backward pass
    beta = backward_algorithm(observations, n_regimes, trans_mat, mu, sigma)
    
    # Compute posterior probabilities (gamma)
    gamma = []
    for t in range(T):
        gamma_t = []
        normalization = sum(alpha[t][j] * beta[t][j] for j in range(n_regimes))
        for j in range(n_regimes):
            if normalization > 0:
                gamma_t.append((alpha[t][j] * beta[t][j]) / normalization)
            else:
                gamma_t.append(1.0 / n_regimes)  # Fallback to uniform distribution
        gamma.append(gamma_t)
    
    return gamma

def viterbi_algorithm(observations, n_regimes, init_probs, trans_mat, mu, sigma):
    """
    Viterbi algorithm to find most likely sequence of hidden states
    
    Returns:
    --------
    path : list
        Most likely sequence of hidden states
    """
    import math
    
    T = len(observations)
    
    # Initialize with log probabilities to avoid underflow
    delta = [[-float('inf') for _ in range(n_regimes)] for _ in range(T)]
    psi = [[0 for _ in range(n_regimes)] for _ in range(T)]
    
    # Initialization
    for j in range(n_regimes):
        try:
            emission_prob = gaussian_pdf(observations[0], mu[j], sigma[j])
            if emission_prob > 0:
                delta[0][j] = math.log(init_probs[j]) + math.log(emission_prob)
            else:
                delta[0][j] = -float('inf')
        except (ValueError, OverflowError):
            delta[0][j] = -float('inf')
    
    # Recursion
    for t in range(1, T):
        for j in range(n_regimes):
            # Find best previous state
            max_val = -float('inf')
            max_idx = 0
            
            for i in range(n_regimes):
                if delta[t-1][i] > -float('inf') and trans_mat[i][j] > 0:
                    val = delta[t-1][i] + math.log(trans_mat[i][j])
                    if val > max_val:
                        max_val = val
                        max_idx = i
            
            # Calculate emission probability
            emission_prob = gaussian_pdf(observations[t], mu[j], sigma[j])
            
            if max_val > -float('inf') and emission_prob > 0:
                delta[t][j] = max_val + math.log(emission_prob)
                psi[t][j] = max_idx
            else:
                delta[t][j] = -float('inf')
                psi[t][j] = 0
    
    # Termination: find the best end state
    max_val = -float('inf')
    max_idx = 0
    for j in range(n_regimes):
        if delta[T-1][j] > max_val:
            max_val = delta[T-1][j]
            max_idx = j
    
    # Backtrack to find the best path
    path = [0] * T
    path[T-1] = max_idx
    
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1][path[t+1]]
    
    return path
