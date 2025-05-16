import random
from .math_utils import gaussian_pdf
import math

def sample_from_prior(n_regimes, mu_prior_mean, mu_prior_std, sigma_prior_alpha, sigma_prior_beta, trans_prior_alpha):
    """Sample initial parameters from prior distributions"""
    # Sample means from Normal prior
    mu = [random.gauss(mu_prior_mean, mu_prior_std) for _ in range(n_regimes)]
    
    # Sample volatilities from Inverse Gamma prior (approximated)
    sigma = []
    for _ in range(n_regimes):
        # Simple approximation of Inverse Gamma
        shape, scale = sigma_prior_alpha, sigma_prior_beta
        gamma_sample = random.gammavariate(shape, scale)
        sigma.append(max(0.001, gamma_sample))  # Ensure positive volatility
        
    # Sample transition matrix rows from Dirichlet prior (approximated)
    trans_mat = []
    for _ in range(n_regimes):
        # Approximate Dirichlet using normalized Gamma samples
        gammas = [random.gammavariate(alpha, 1.0) for alpha in trans_prior_alpha]
        total = sum(gammas)
        if total > 0:
            trans_mat.append([g/total for g in gammas])
        else:
            # Fallback to uniform if sampling fails
            trans_mat.append([1.0/n_regimes] * n_regimes)
    
    # Initial probabilities (uniform)
    init_probs = [1.0/n_regimes] * n_regimes
    
    return mu, sigma, trans_mat, init_probs

def initialize_kmeans(observations, n_regimes):
    """Initialize parameters using k-means like approach"""
    # Simple implementation of k-means for initialization
    n = len(observations)
    indices = random.sample(range(n), min(n_regimes, n))
    centroids = [observations[i] for i in indices]
    
    # Assign observations to nearest centroid
    assignments = [0] * n
    for i in range(n):
        min_dist = float('inf')
        min_idx = 0
        for j in range(len(centroids)):
            dist = (observations[i] - centroids[j]) ** 2
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        assignments[i] = min_idx
    
    # Calculate means and variances
    mu = [0.0] * n_regimes
    sigma = [0.01] * n_regimes  # Default to small positive value
    counts = [0] * n_regimes
    
    for i in range(n):
        cluster = assignments[i]
        mu[cluster] += observations[i]
        counts[cluster] += 1
    
    # Calculate means
    for j in range(n_regimes):
        if counts[j] > 0:
            mu[j] /= counts[j]
    
    # Calculate standard deviations
    for i in range(n):
        cluster = assignments[i]
        sigma[cluster] += (observations[i] - mu[cluster]) ** 2
    
    for j in range(n_regimes):
        if counts[j] > 1:
            sigma[j] = math.sqrt(sigma[j] / counts[j])
        else:
            # Use a small positive value for empty clusters
            sigma[j] = 0.01
    
    # Initialize transition matrix with high self-transition probability
    trans_mat = []
    for i in range(n_regimes):
        row = []
        for j in range(n_regimes):
            if i == j:
                row.append(0.8)  # High probability of staying in same state
            else:
                row.append(0.2 / (n_regimes - 1))  # Distribute remaining probability
        trans_mat.append(row)
    
    # Initial probabilities based on cluster sizes
    init_probs = [count / n for count in counts]
    
    # Handle empty clusters
    for j in range(n_regimes):
        if init_probs[j] == 0:
            init_probs[j] = 0.01
    
    # Normalize initial probabilities
    total = sum(init_probs)
    init_probs = [p / total for p in init_probs]
    
    return mu, sigma, trans_mat, init_probs
