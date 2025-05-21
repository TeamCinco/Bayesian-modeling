import math
from .math_utils import gaussian_pdf, log_sum_exp
from .algorithms import forward_algorithm, backward_algorithm

def compute_xi(observations, n_regimes, trans_mat, mu, sigma, alpha, beta):
    """Compute joint probabilities xi"""
    T = len(observations)
    xi = [[[0.0 for _ in range(n_regimes)] for _ in range(n_regimes)] for _ in range(T-1)]
    
    for t in range(T-1):
        # Compute denominator (normalization factor)
        denominator = 0.0
        for i in range(n_regimes):
            for j in range(n_regimes):
                emission_prob = gaussian_pdf(observations[t+1], mu[j], sigma[j])
                denominator += alpha[t][i] * trans_mat[i][j] * emission_prob * beta[t+1][j]
        
        # Compute xi
        for i in range(n_regimes):
            for j in range(n_regimes):
                if denominator > 0:
                    emission_prob = gaussian_pdf(observations[t+1], mu[j], sigma[j])
                    xi[t][i][j] = (alpha[t][i] * trans_mat[i][j] * emission_prob * beta[t+1][j]) / denominator
                else:
                    xi[t][i][j] = 1.0 / (n_regimes * n_regimes)  # Uniform if underflow
    
    return xi

def update_parameters(observations, gamma, xi, n_regimes, 
                     mu_prior_mean, mu_prior_std, sigma_prior_alpha, sigma_prior_beta, 
                     trans_prior_alpha):
    """Update parameters with Bayesian priors (MAP estimation)"""
    T = len(observations)
    
    # Initialize parameter containers
    mu = [0.0] * n_regimes
    sigma = [0.0] * n_regimes
    trans_mat = [[0.0 for _ in range(n_regimes)] for _ in range(n_regimes)]
    init_probs = [0.0] * n_regimes
    
    # Update initial probabilities (with Dirichlet prior)
    prior_count = 1.0  # Strength of prior
    for j in range(n_regimes):
        init_probs[j] = (gamma[0][j] + prior_count - 1) / (1 + n_regimes * (prior_count - 1))
    
    # Update transition matrix (with Dirichlet prior)
    for i in range(n_regimes):
        denominator = sum(sum(xi[t][i][j] for j in range(n_regimes)) for t in range(T-1))
        denominator += sum(trans_prior_alpha) - n_regimes  # Add prior - 1 for each element
        
        for j in range(n_regimes):
            numerator = sum(xi[t][i][j] for t in range(T-1))
            numerator += trans_prior_alpha[j] - 1  # Add prior - 1
            
            if denominator > 0:
                trans_mat[i][j] = numerator / denominator
            else:
                trans_mat[i][j] = 1.0 / n_regimes
    
    # Update output parameters (mu and sigma) with Gaussian and InvGamma priors
    for j in range(n_regimes):
        # Count of times in state j (weighted by gamma)
        state_count = sum(gamma[t][j] for t in range(T))
        
        if state_count > 0:
            # Update mu (with Normal prior)
            prior_precision = 1.0 / (mu_prior_std ** 2)
            data_precision = state_count / (sigma_prior_beta ** 2)  # Using prior as initial value
            
            # Weighted sum of observations in state j
            weighted_sum = sum(gamma[t][j] * observations[t] for t in range(T))
            
            # MAP estimate for mu
            posterior_precision = prior_precision + data_precision
            posterior_mean = (prior_precision * mu_prior_mean + data_precision * (weighted_sum / state_count)) / posterior_precision
            
            mu[j] = posterior_mean
            
            # Update sigma (with InvGamma prior)
            weighted_var_sum = sum(gamma[t][j] * (observations[t] - mu[j]) ** 2 for t in range(T))
            
            # MAP estimate for sigma (approximation)
            posterior_alpha = sigma_prior_alpha + state_count / 2
            posterior_beta = sigma_prior_beta + weighted_var_sum / 2
            
            if posterior_alpha > 1:  # Ensure mode exists
                sigma[j] = math.sqrt(posterior_beta / (posterior_alpha + 1))
            else:
                sigma[j] = math.sqrt(posterior_beta / 2)  # Fallback
        else:
            # Default values if state never observed
            mu[j] = mu_prior_mean
            sigma[j] = math.sqrt(sigma_prior_beta)
        
        # Ensure sigma is positive
        sigma[j] = max(0.0001, sigma[j])
    
    return mu, sigma, trans_mat, init_probs

def compute_log_likelihood(observations, n_regimes, init_probs, trans_mat, mu, sigma):
    """Compute log likelihood of observations given current parameters"""
    # Use forward algorithm alpha values for the last time step
    alpha = forward_algorithm(observations, n_regimes, init_probs, trans_mat, mu, sigma)
    T = len(observations)
    
    # Log likelihood is log of sum of alpha values at final time step
    ll = -float('inf')
    for j in range(n_regimes):
        ll = log_sum_exp(ll, math.log(max(1e-300, alpha[T-1][j])))
    
    return ll
