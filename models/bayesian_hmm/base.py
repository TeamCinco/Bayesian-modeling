import random
import math
from .math_utils import gaussian_pdf, log_sum_exp
from .initialization import sample_from_prior, initialize_kmeans
from .algorithms import forward_algorithm, backward_algorithm, forward_backward_algorithm, viterbi_algorithm
from .parameter_estimation import compute_xi, update_parameters, compute_log_likelihood
from .prediction import predict_states, predict_probabilities, forecast_regime_proba

class BayesianHMM:
    def __init__(self, n_regimes=3, seed=42):
        """
        Initialize Bayesian Hidden Markov Model for market regime detection
        
        Parameters:
        -----------
        n_regimes : int
            Number of market regimes to identify (default: 3)
        seed : int
            Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.seed = seed
        random.seed(seed)
        
        # These will be set during fit
        self.mu = None          # Mean returns per regime
        self.sigma = None       # Volatility per regime
        self.trans_mat = None   # Transition matrix between regimes
        self.init_probs = None  # Initial regime probabilities
        
        # Priors (Bayesian component)
        self.mu_prior_mean = 0.0
        self.mu_prior_std = 0.01
        self.sigma_prior_alpha = 2.0
        self.sigma_prior_beta = 0.001
        self.trans_prior_alpha = [1.0] * n_regimes  # Dirichlet prior

    def fit(self, observations, max_iter=100, tol=1e-6, initialize='kmeans'):
        """
        Fit model parameters using Baum-Welch algorithm with Bayesian priors
        
        Parameters:
        -----------
        observations : list
            Time series of observations (e.g., returns)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        initialize : str
            Initialization method ('kmeans' or 'random')
            
        Returns:
        --------
        self : object
            Returns self
        """
        if len(observations) < 10:
            raise ValueError("Not enough observations to fit model")
        
        # Initialize parameters
        if initialize == 'random':
            self.mu, self.sigma, self.trans_mat, self.init_probs = sample_from_prior(
                self.n_regimes, self.mu_prior_mean, self.mu_prior_std,
                self.sigma_prior_alpha, self.sigma_prior_beta, self.trans_prior_alpha
            )
        else:  # kmeans-like initialization
            self.mu, self.sigma, self.trans_mat, self.init_probs = initialize_kmeans(
                observations, self.n_regimes
            )
        
        # Run Baum-Welch algorithm (EM for HMM)
        prev_log_likelihood = -float('inf')
        
        for iteration in range(max_iter):
            # E-step: Compute posterior probabilities
            gamma = forward_backward_algorithm(
                observations, self.n_regimes, self.init_probs,
                self.trans_mat, self.mu, self.sigma
            )
            
            # Compute forward and backward probabilities once
            alpha = forward_algorithm(
                observations, self.n_regimes, self.init_probs,
                self.trans_mat, self.mu, self.sigma
            )
            beta = backward_algorithm(
                observations, self.n_regimes, self.trans_mat, self.mu, self.sigma
            )
            
            # Compute xi (joint probability of being in state i at t and state j at t+1)
            xi = compute_xi(
                observations, self.n_regimes, self.trans_mat,
                self.mu, self.sigma, alpha, beta
            )
            
            # M-step: Update parameters with Bayesian priors
            self.mu, self.sigma, self.trans_mat, self.init_probs = update_parameters(
                observations, gamma, xi, self.n_regimes,
                self.mu_prior_mean, self.mu_prior_std,
                self.sigma_prior_alpha, self.sigma_prior_beta,
                self.trans_prior_alpha
            )
            
            # Compute log likelihood
            log_likelihood = compute_log_likelihood(
                observations, self.n_regimes, self.init_probs,
                self.trans_mat, self.mu, self.sigma
            )
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            
            prev_log_likelihood = log_likelihood
        
        return self

    def predict(self, observations):
        """
        Predict most likely regime sequence for observations
        
        Parameters:
        -----------
        observations : list
            Time series of observations
            
        Returns:
        --------
        states : list
            Most likely hidden state sequence
        """
        return predict_states(
            observations, self.n_regimes, self.init_probs,
            self.trans_mat, self.mu, self.sigma
        )
    
    def predict_proba(self, observations):
        """
        Predict regime probabilities for observations
        
        Parameters:
        -----------
        observations : list
            Time series of observations
            
        Returns:
        --------
        probabilities : list of lists
            Probability of each regime at each time step
        """
        return predict_probabilities(
            observations, self.n_regimes, self.init_probs,
            self.trans_mat, self.mu, self.sigma
        )
    
    def forecast_regime_proba(self, current_regime, steps=1):
        """
        Forecast regime probabilities n steps ahead
        
        Parameters:
        -----------
        current_regime : int
            Current regime index
        steps : int
            Number of steps to forecast ahead
            
        Returns:
        --------
        probabilities : list
            Probability of each regime after n steps
        """
        return forecast_regime_proba(
            current_regime, steps, self.n_regimes, self.trans_mat
        )
    
    # For backward compatibility and direct access, include these methods
    def gaussian_pdf(self, x, mu, sigma):
        """Compute Gaussian probability density function"""
        return gaussian_pdf(x, mu, sigma)
    
    def forward_backward(self, observations):
        """Run forward-backward algorithm on observation sequence"""
        return forward_backward_algorithm(
            observations, self.n_regimes, self.init_probs,
            self.trans_mat, self.mu, self.sigma
        )
    
    def viterbi(self, observations):
        """Run Viterbi algorithm to find most likely sequence of hidden states"""
        return viterbi_algorithm(
            observations, self.n_regimes, self.init_probs,
            self.trans_mat, self.mu, self.sigma
        )
    
    @property
    def transmat_(self):
        """
        Property that returns the transition matrix.
        This is to maintain compatibility with code that expects a 'transmat_' attribute.
        """
        return self.trans_mat
    
    @property
    def startprob_(self):
        """
        Property that returns the initial state probabilities.
        This is to maintain compatibility with code that expects a 'startprob_' attribute.
        """
        return self.init_probs
