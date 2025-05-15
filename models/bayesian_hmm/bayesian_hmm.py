import math
import random

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

    def _sample_from_prior(self):
        """Sample initial parameters from prior distributions"""
        # Sample means from Normal prior
        mu = [random.gauss(self.mu_prior_mean, self.mu_prior_std) for _ in range(self.n_regimes)]
        
        # Sample volatilities from Inverse Gamma prior (approximated)
        # For simplicity, we're using a Gamma and taking reciprocal
        sigma = []
        for _ in range(self.n_regimes):
            # Simple approximation of Inverse Gamma
            shape, scale = self.sigma_prior_alpha, self.sigma_prior_beta
            gamma_sample = random.gammavariate(shape, scale)
            sigma.append(max(0.001, gamma_sample))  # Ensure positive volatility
            
        # Sample transition matrix rows from Dirichlet prior (approximated using Gamma)
        trans_mat = []
        for _ in range(self.n_regimes):
            # Simple approximation of Dirichlet using normalized Gamma samples
            gammas = [random.gammavariate(alpha, 1.0) for alpha in self.trans_prior_alpha]
            total = sum(gammas)
            if total > 0:
                trans_mat.append([g/total for g in gammas])
            else:
                # Fallback to uniform if sampling fails
                trans_mat.append([1.0/self.n_regimes] * self.n_regimes)
        
        # Initial probabilities (uniform)
        init_probs = [1.0/self.n_regimes] * self.n_regimes
        
        return mu, sigma, trans_mat, init_probs

    def gaussian_pdf(self, x, mu, sigma):
        """Compute Gaussian probability density function"""
        if sigma <= 0:
            return 0.0
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        try:
            return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)
        except OverflowError:
            # Handle numerical underflow
            if exponent < -700:  # Approximate lower bound for exp in Python
                return 1e-300  # Small positive number instead of zero
            return 0.0

    def forward_backward(self, observations):
        """
        Run forward-backward algorithm on observation sequence
        
        Parameters:
        -----------
        observations : list
            Time series of observations (e.g., returns or features)
            
        Returns:
        --------
        gamma : list of lists
            Posterior probabilities P(S_t=j|observations) for each time step and regime
        """
        T = len(observations)
        
        # Forward pass
        alpha = self._forward(observations)
        
        # Backward pass
        beta = self._backward(observations)
        
        # Compute posterior probabilities (gamma)
        gamma = []
        for t in range(T):
            gamma_t = []
            normalization = sum(alpha[t][j] * beta[t][j] for j in range(self.n_regimes))
            for j in range(self.n_regimes):
                if normalization > 0:
                    gamma_t.append((alpha[t][j] * beta[t][j]) / normalization)
                else:
                    gamma_t.append(1.0 / self.n_regimes)  # Fallback to uniform distribution
            gamma.append(gamma_t)
        
        return gamma

    def _forward(self, observations):
        """Forward algorithm for computing alpha variables"""
        T = len(observations)
        alpha = [[0.0 for _ in range(self.n_regimes)] for _ in range(T)]
        
        # Initialize
        for j in range(self.n_regimes):
            alpha[0][j] = self.init_probs[j] * self.gaussian_pdf(observations[0], self.mu[j], self.sigma[j])
        
        # Normalize to prevent underflow
        alpha_sum = sum(alpha[0])
        if alpha_sum > 0:
            for j in range(self.n_regimes):
                alpha[0][j] /= alpha_sum
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_regimes):
                alpha[t][j] = sum(alpha[t-1][i] * self.trans_mat[i][j] for i in range(self.n_regimes))
                alpha[t][j] *= self.gaussian_pdf(observations[t], self.mu[j], self.sigma[j])
            
            # Normalize
            alpha_sum = sum(alpha[t])
            if alpha_sum > 0:
                for j in range(self.n_regimes):
                    alpha[t][j] /= alpha_sum
        
        return alpha

    def _backward(self, observations):
        """Backward algorithm for computing beta variables"""
        T = len(observations)
        beta = [[0.0 for _ in range(self.n_regimes)] for _ in range(T)]
        
        # Initialize
        for j in range(self.n_regimes):
            beta[T-1][j] = 1.0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_regimes):
                beta[t][i] = 0.0
                for j in range(self.n_regimes):
                    beta[t][i] += self.trans_mat[i][j] * self.gaussian_pdf(observations[t+1], self.mu[j], self.sigma[j]) * beta[t+1][j]
            
            # Normalize
            beta_sum = sum(beta[t])
            if beta_sum > 0:
                for i in range(self.n_regimes):
                    beta[t][i] /= beta_sum
        
        return beta

    def viterbi(self, observations):
        """
        Viterbi algorithm to find most likely sequence of hidden states
        
        Parameters:
        -----------
        observations : list
            Time series of observations
            
        Returns:
        --------
        path : list
            Most likely sequence of hidden states
        """
        T = len(observations)
        
        # Initialize with log probabilities to avoid underflow
        delta = [[-float('inf') for _ in range(self.n_regimes)] for _ in range(T)]
        psi = [[0 for _ in range(self.n_regimes)] for _ in range(T)]
        
        # Initialization
        for j in range(self.n_regimes):
            try:
                emission_prob = self.gaussian_pdf(observations[0], self.mu[j], self.sigma[j])
                if emission_prob > 0:
                    delta[0][j] = math.log(self.init_probs[j]) + math.log(emission_prob)
                else:
                    delta[0][j] = -float('inf')
            except (ValueError, OverflowError):
                delta[0][j] = -float('inf')
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_regimes):
                # Find best previous state
                max_val = -float('inf')
                max_idx = 0
                
                for i in range(self.n_regimes):
                    if delta[t-1][i] > -float('inf') and self.trans_mat[i][j] > 0:
                        val = delta[t-1][i] + math.log(self.trans_mat[i][j])
                        if val > max_val:
                            max_val = val
                            max_idx = i
                
                # Calculate emission probability
                emission_prob = self.gaussian_pdf(observations[t], self.mu[j], self.sigma[j])
                
                if max_val > -float('inf') and emission_prob > 0:
                    delta[t][j] = max_val + math.log(emission_prob)
                    psi[t][j] = max_idx
                else:
                    delta[t][j] = -float('inf')
                    psi[t][j] = 0
        
        # Termination: find the best end state
        max_val = -float('inf')
        max_idx = 0
        for j in range(self.n_regimes):
            if delta[T-1][j] > max_val:
                max_val = delta[T-1][j]
                max_idx = j
        
        # Backtrack to find the best path
        path = [0] * T
        path[T-1] = max_idx
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1][path[t+1]]
        
        return path

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
            self.mu, self.sigma, self.trans_mat, self.init_probs = self._sample_from_prior()
        else:  # kmeans-like initialization
            self._initialize_kmeans(observations)
        
        # Run Baum-Welch algorithm (EM for HMM)
        prev_log_likelihood = -float('inf')
        
        for iteration in range(max_iter):
            # E-step: Compute posterior probabilities
            gamma = self.forward_backward(observations)
            
            # Compute xi (joint probability of being in state i at t and state j at t+1)
            xi = self._compute_xi(observations, gamma)
            
            # M-step: Update parameters with Bayesian priors
            self._update_parameters(observations, gamma, xi)
            
            # Compute log likelihood
            log_likelihood = self._compute_log_likelihood(observations)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            
            prev_log_likelihood = log_likelihood
        
        return self

    def _initialize_kmeans(self, observations):
        """Initialize parameters using k-means like approach"""
        # Simple implementation of k-means for initialization
        # 1. Pick k random observations as initial centroids
        n = len(observations)
        indices = random.sample(range(n), min(self.n_regimes, n))
        centroids = [observations[i] for i in indices]
        
        # 2. Assign observations to nearest centroid
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
        
        # 3. Calculate means and variances for each cluster
        self.mu = [0.0] * self.n_regimes
        self.sigma = [0.01] * self.n_regimes  # Default to small positive value
        counts = [0] * self.n_regimes
        
        for i in range(n):
            cluster = assignments[i]
            self.mu[cluster] += observations[i]
            counts[cluster] += 1
        
        # Calculate means
        for j in range(self.n_regimes):
            if counts[j] > 0:
                self.mu[j] /= counts[j]
        
        # Calculate standard deviations
        for i in range(n):
            cluster = assignments[i]
            self.sigma[cluster] += (observations[i] - self.mu[cluster]) ** 2
        
        for j in range(self.n_regimes):
            if counts[j] > 1:
                self.sigma[j] = math.sqrt(self.sigma[j] / counts[j])
            else:
                # Use a small positive value for empty clusters
                self.sigma[j] = 0.01
        
        # Initialize transition matrix with high self-transition probability
        self.trans_mat = []
        for i in range(self.n_regimes):
            row = []
            for j in range(self.n_regimes):
                if i == j:
                    row.append(0.8)  # High probability of staying in same state
                else:
                    row.append(0.2 / (self.n_regimes - 1))  # Distribute remaining probability
            self.trans_mat.append(row)
        
        # Initial probabilities based on cluster sizes
        self.init_probs = [count / n for count in counts]
        
        # Handle empty clusters
        for j in range(self.n_regimes):
            if self.init_probs[j] == 0:
                self.init_probs[j] = 0.01
        
        # Normalize initial probabilities
        total = sum(self.init_probs)
        self.init_probs = [p / total for p in self.init_probs]

    def _compute_xi(self, observations, gamma):
        """Compute joint probabilities xi"""
        T = len(observations)
        xi = [[[0.0 for _ in range(self.n_regimes)] for _ in range(self.n_regimes)] for _ in range(T-1)]
        
        # Compute forward and backward probabilities once
        alpha = self._forward(observations)
        beta = self._backward(observations)
        
        for t in range(T-1):
            # Compute denominator (normalization factor)
            denominator = 0.0
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    emission_prob = self.gaussian_pdf(observations[t+1], self.mu[j], self.sigma[j])
                    denominator += alpha[t][i] * self.trans_mat[i][j] * emission_prob * beta[t+1][j]
            
            # Compute xi
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    if denominator > 0:
                        emission_prob = self.gaussian_pdf(observations[t+1], self.mu[j], self.sigma[j])
                        xi[t][i][j] = (alpha[t][i] * self.trans_mat[i][j] * emission_prob * beta[t+1][j]) / denominator
                    else:
                        xi[t][i][j] = 1.0 / (self.n_regimes * self.n_regimes)  # Uniform if underflow
        
        return xi

    def _update_parameters(self, observations, gamma, xi):
        """Update parameters with Bayesian priors (MAP estimation)"""
        T = len(observations)
        
        # Update initial probabilities (with Dirichlet prior)
        prior_count = 1.0  # Strength of prior
        for j in range(self.n_regimes):
            self.init_probs[j] = (gamma[0][j] + prior_count - 1) / (1 + self.n_regimes * (prior_count - 1))
        
        # Update transition matrix (with Dirichlet prior)
        for i in range(self.n_regimes):
            denominator = sum(sum(xi[t][i][j] for j in range(self.n_regimes)) for t in range(T-1))
            denominator += sum(self.trans_prior_alpha) - self.n_regimes  # Add prior - 1 for each element
            
            for j in range(self.n_regimes):
                numerator = sum(xi[t][i][j] for t in range(T-1))
                numerator += self.trans_prior_alpha[j] - 1  # Add prior - 1
                
                if denominator > 0:
                    self.trans_mat[i][j] = numerator / denominator
                else:
                    self.trans_mat[i][j] = 1.0 / self.n_regimes
        
        # Update output parameters (mu and sigma) with Gaussian and InvGamma priors
        for j in range(self.n_regimes):
            # Count of times in state j (weighted by gamma)
            state_count = sum(gamma[t][j] for t in range(T))
            
            if state_count > 0:
                # Update mu (with Normal prior)
                prior_precision = 1.0 / (self.mu_prior_std ** 2)
                data_precision = state_count / (self.sigma[j] ** 2)
                
                # Weighted sum of observations in state j
                weighted_sum = sum(gamma[t][j] * observations[t] for t in range(T))
                
                # MAP estimate for mu
                posterior_precision = prior_precision + data_precision
                posterior_mean = (prior_precision * self.mu_prior_mean + data_precision * (weighted_sum / state_count)) / posterior_precision
                
                self.mu[j] = posterior_mean
                
                # Update sigma (with InvGamma prior)
                weighted_var_sum = sum(gamma[t][j] * (observations[t] - self.mu[j]) ** 2 for t in range(T))
                
                # MAP estimate for sigma (approximation)
                posterior_alpha = self.sigma_prior_alpha + state_count / 2
                posterior_beta = self.sigma_prior_beta + weighted_var_sum / 2
                
                if posterior_alpha > 1:  # Ensure mode exists
                    self.sigma[j] = math.sqrt(posterior_beta / (posterior_alpha + 1))
                else:
                    self.sigma[j] = math.sqrt(posterior_beta / 2)  # Fallback
            
            # Ensure sigma is positive
            self.sigma[j] = max(0.0001, self.sigma[j])

    def _compute_log_likelihood(self, observations):
        """Compute log likelihood of observations given current parameters"""
        # Use forward algorithm alpha values for the last time step
        alpha = self._forward(observations)
        T = len(observations)
        
        # Log likelihood is log of sum of alpha values at final time step
        ll = -float('inf')
        for j in range(self.n_regimes):
            ll = self._log_sum_exp(ll, math.log(max(1e-300, alpha[T-1][j])))
        
        return ll

    def _log_sum_exp(self, a, b):
        """Numerically stable implementation of log(exp(a) + exp(b))"""
        if a == -float('inf'):
            return b
        if b == -float('inf'):
            return a
        if a > b:
            return a + math.log(1 + math.exp(b - a))
        else:
            return b + math.log(1 + math.exp(a - b))

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
        return self.viterbi(observations)
    
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
        return self.forward_backward(observations)
    
    @property
    def transmat_(self):
        """
        Property that returns the transition matrix.
        This is to maintain compatibility with code that expects a 'transmat_' attribute.
        """
        return self.trans_mat
    
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
        probs = [0.0] * self.n_regimes
        probs[current_regime] = 1.0
        
        # Apply transition matrix 'steps' times
        for _ in range(steps):
            new_probs = [0.0] * self.n_regimes
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    new_probs[j] += probs[i] * self.trans_mat[i][j]
            probs = new_probs
        
        return probs
