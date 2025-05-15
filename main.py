import os
import math
import csv
from datetime import datetime, timedelta
import random
import glob as glob
import re
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


class FeatureExtractor:
    """Extract features from OHLCV data for regime detection"""
    
    @staticmethod
    def log_returns(prices):
        """Calculate log returns from price series"""
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] <= 0 or prices[i] <= 0:
                returns.append(0.0)  # Handle invalid prices
            else:
                returns.append(math.log(prices[i] / prices[i-1]))
        return returns
    
    @staticmethod
    def rolling_volatility(returns, window=5):
        """Calculate rolling volatility (standard deviation)"""
        volatility = []
        for i in range(len(returns)):
            if i < window - 1:
                # Not enough data for full window, use available data
                window_returns = returns[:i+1]
            else:
                window_returns = returns[i-window+1:i+1]
            
            # Calculate std dev if enough data
            if len(window_returns) > 1:
                mean = sum(window_returns) / len(window_returns)
                variance = sum((r - mean) ** 2 for r in window_returns) / len(window_returns)
                volatility.append(math.sqrt(variance))
            else:
                volatility.append(0.0)
        
        return volatility
    
    @staticmethod
    def atr(high, low, close, window=14):
        """Calculate Average True Range"""
        tr = []  # True Range
        
        # Calculate True Range
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr.append(max(tr1, tr2, tr3))
        
        # Calculate ATR with Simple Moving Average
        atr = []
        for i in range(len(tr)):
            if i < window:
                # Use simple average for initial values
                atr.append(sum(tr[:i+1]) / (i+1))
            else:
                # Use simple moving average
                atr.append(sum(tr[i-window+1:i+1]) / window)
        
        # Add a 0 for the first day (no TR calculated)
        return [0.0] + atr
    
    @staticmethod
    def bollinger_width(close, window=20, num_std=2):
        """Calculate Bollinger Band width (normalized)"""
        bb_width = []
        
        for i in range(len(close)):
            if i < window - 1:
                # Not enough data, use simple estimate
                mean = sum(close[:i+1]) / (i+1)
                if i > 0:
                    std = math.sqrt(sum((p - mean) ** 2 for p in close[:i+1]) / (i+1))
                else:
                    std = 0.0
            else:
                window_prices = close[i-window+1:i+1]
                mean = sum(window_prices) / window
                std = math.sqrt(sum((p - mean) ** 2 for p in window_prices) / window)
            
            # Calculate width and normalize by mean price
            width = (2 * num_std * std)
            if mean > 0:
                bb_width.append(width / mean)  # Normalized by price level
            else:
                bb_width.append(0.0)
        
        return bb_width
    
    @staticmethod
    def momentum(close, short_window=10, long_window=20):
        """Calculate price momentum relative to moving averages"""
        momentum = []
        
        for i in range(len(close)):
            if i < short_window - 1:
                # Not enough data for short MA
                momentum.append(0.0)
            elif i < long_window - 1:
                # Only short MA available
                short_ma = sum(close[i-short_window+1:i+1]) / short_window
                momentum.append(close[i] / short_ma - 1.0)
            else:
                # Both MAs available
                short_ma = sum(close[i-short_window+1:i+1]) / short_window
                long_ma = sum(close[i-long_window+1:i+1]) / long_window
                # Ratio of short MA to long MA
                momentum.append(short_ma / long_ma - 1.0)
        
        return momentum
    
    @staticmethod
    def rsi(close, window=14):
        """Calculate Relative Strength Index"""
        rsi = []
        
        # Need at least 2 prices to calculate gains/losses
        if len(close) < 2:
            return [50.0] * len(close)
        
        # Calculate price changes
        changes = [close[i] - close[i-1] for i in range(1, len(close))]
        
        # Separate gains and losses
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        # Calculate RSI
        for i in range(len(changes)):
            if i < window:
                # Use simple averages for first window periods
                avg_gain = sum(gains[:i+1]) / (i+1)
                avg_loss = sum(losses[:i+1]) / (i+1)
            else:
                # Use wilder's smoothing
                avg_gain = (gains[i] + (window - 1) * avg_gain) / window
                avg_loss = (losses[i] + (window - 1) * avg_loss) / window
            
            if avg_loss == 0:
                rsi.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
        
        # Add a value for the first day
        return [50.0] + rsi
    
    @staticmethod
    def extract_features(ohlcv_data, config=None):
        """
        Extract all features from OHLCV data
        
        Parameters:
        -----------
        ohlcv_data : dict
            Dictionary with 'open', 'high', 'low', 'close', 'volume' keys
        config : dict
            Feature configuration parameters
            
        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        # Default config
        if config is None:
            config = {
                'volatility_window': 5,
                'atr_window': 14,
                'bb_window': 20,
                'momentum_short': 10,
                'momentum_long': 20,
                'rsi_window': 14
            }
        
        # Extract individual features
        features = {}
        features['log_returns'] = FeatureExtractor.log_returns(ohlcv_data['close'])
        features['volatility'] = FeatureExtractor.rolling_volatility(
            features['log_returns'], window=config['volatility_window']
        )
        features['atr'] = FeatureExtractor.atr(
            ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'], 
            window=config['atr_window']
        )
        features['bb_width'] = FeatureExtractor.bollinger_width(
            ohlcv_data['close'], window=config['bb_window']
        )
        features['momentum'] = FeatureExtractor.momentum(
            ohlcv_data['close'], short_window=config['momentum_short'], long_window=config['momentum_long']
        )
        features['rsi'] = FeatureExtractor.rsi(
            ohlcv_data['close'], window=config['rsi_window']
        )
        
        return features

# Modify the load_csv_data function to handle your specific file format
def load_csv_data(file_path):
    """Load OHLCV data from CSV file"""
    dates = []
    ohlcv = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle the datetime format in your files
            date_str = row.get('datetime', '')
            if date_str:
                try:
                    # Parse datetime and extract date portion
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    dates.append(date_obj.date())
                except ValueError:
                    # Try alternative format if first one fails
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        dates.append(date_obj.date())
                    except ValueError:
                        continue  # Skip rows with invalid dates
            
            try:
                ohlcv['open'].append(float(row['open']))
                ohlcv['high'].append(float(row['high']))
                ohlcv['low'].append(float(row['low']))
                ohlcv['close'].append(float(row['close']))
                ohlcv['volume'].append(float(row['volume']))
            except (ValueError, KeyError) as e:
                # Handle missing data by using previous values or default
                if ohlcv['open']:  # If we have previous values
                    ohlcv['open'].append(ohlcv['open'][-1])
                    ohlcv['high'].append(ohlcv['high'][-1])
                    ohlcv['low'].append(ohlcv['low'][-1])
                    ohlcv['close'].append(ohlcv['close'][-1])
                    ohlcv['volume'].append(0.0)
                else:  # No previous values
                    ohlcv['open'].append(0.0)
                    ohlcv['high'].append(0.0)
                    ohlcv['low'].append(0.0)
                    ohlcv['close'].append(0.0)
                    ohlcv['volume'].append(0.0)
    
    return dates, ohlcv

def normalize_features(features):
   """Normalize features to have 0 mean and unit variance"""
   normalized = {}
   
   for name, values in features.items():
       # Calculate mean and std
       valid_values = [v for v in values if v is not None and not math.isnan(v)]
       if not valid_values:
           normalized[name] = [0.0] * len(values)
           continue
           
       mean = sum(valid_values) / len(valid_values)
       variance = sum((v - mean) ** 2 for v in valid_values) / len(valid_values)
       std = math.sqrt(variance) if variance > 0 else 1.0
       
       # Normalize values
       normalized[name] = [(v - mean) / std if v is not None and not math.isnan(v) else 0.0 for v in values]
   
   return normalized


def combine_features(features, weights=None):
   """
   Combine multiple features into a single series
   
   Parameters:
   -----------
   features : dict
       Dictionary of feature arrays
   weights : dict
       Dictionary of weights for each feature
       
   Returns:
   --------
   combined : list
       Combined feature series
   """
   if weights is None:
       # Default equal weights
       weights = {name: 1.0 for name in features}
   
   # Ensure all features have the same length
   min_length = min(len(values) for values in features.values())
   
   # Initialize combined series
   combined = [0.0] * min_length
   
   # Compute weighted sum
   total_weight = sum(weights.values())
   for name, values in features.items():
       if name in weights and weights[name] > 0:
           weight = weights[name] / total_weight
           for i in range(min_length):
               combined[i] += values[i] * weight
   
   return combined


def analyze_market_regimes(file_path, n_regimes=3, feature_config=None, feature_weights=None):
   """
   Analyze market regimes using Bayesian HMM
   
   Parameters:
   -----------
   file_path : str
       Path to CSV file with OHLCV data
   n_regimes : int
       Number of regimes to identify
   feature_config : dict
       Configuration parameters for feature extraction
   feature_weights : dict
       Weights for combining features
       
   Returns:
   --------
   results : dict
       Analysis results including regime assignments and model
   """
   print(f"Analyzing market regimes with {n_regimes} states...")
   
   # Load data
   dates, ohlcv = load_csv_data(file_path)
   
   if len(dates) < 100:
       raise ValueError(f"Not enough data points: {len(dates)}")
   
   # Extract features
   features = FeatureExtractor.extract_features(ohlcv, config=feature_config)
   
   # Normalize features
   normalized_features = normalize_features(features)
   
   # Combine features into a single series
   combined_feature = combine_features(normalized_features, weights=feature_weights)
   
   # Fit Bayesian HMM
   hmm = BayesianHMM(n_regimes=n_regimes)
   hmm.fit(combined_feature, max_iter=100)
   
   # Get regime assignments
   regimes = hmm.predict(combined_feature)
   
   # Get regime probabilities
   regime_probs = hmm.predict_proba(combined_feature)
   
   # Analyze regime characteristics
   regime_stats = analyze_regime_statistics(regimes, features, ohlcv, dates)
   
   # Sort regimes by volatility for better interpretation
   sorted_regimes, regime_mapping = sort_regimes_by_volatility(regimes, regime_stats)
   
   # Map regime probabilities to sorted regimes
   sorted_regime_probs = []
   for t in range(len(regime_probs)):
       probs = [0.0] * n_regimes
       for old_regime, new_regime in regime_mapping.items():
           probs[new_regime] = regime_probs[t][old_regime]
       sorted_regime_probs.append(probs)
   
   # Create results dictionary
   results = {
       'dates': dates,
       'ohlcv': ohlcv,
       'features': features,
       'regimes': sorted_regimes,
       'regime_probs': sorted_regime_probs,
       'regime_stats': regime_stats,
       'hmm_model': hmm,
       'regime_mapping': regime_mapping
   }
   
   # Print regime statistics
   print("\nMarket Regime Statistics:")
   for i, stat in enumerate(sorted(regime_stats, key=lambda x: x['volatility'])):
       print(f"Regime {i+1} (was {stat['regime']+1}):")
       print(f"  Count: {stat['count']} days ({stat['percentage']:.1f}%)")
       print(f"  Avg Return: {stat['mean_return']*100:.2f}% daily")
       print(f"  Volatility: {stat['volatility']*100:.2f}%")
       print(f"  Sharpe: {stat['sharpe']:.2f}")
       print(f"  Average ATR: {stat['mean_atr']:.4f}")
       print(f"  Period: {stat['start_date']} to {stat['end_date']}")
       print(f"  Description: {stat['description']}")
       print()
   
   return results


def analyze_regime_statistics(regimes, features, ohlcv, dates):
   """Calculate statistics for each regime"""
   n_regimes = max(regimes) + 1
   stats = []
   
   # Extract needed features
   returns = features['log_returns']
   volatility = features['volatility']
   atr = features['atr']
   
   for regime in range(n_regimes):
       # Find indices for this regime
       indices = [i for i in range(len(regimes)) if regimes[i] == regime]
       
       if not indices:
           continue
       
       # Collect regime data
       regime_returns = [returns[i] for i in indices]
       regime_volatility = [volatility[i] for i in indices]
       regime_atr = [atr[i] for i in indices]
       
       # Calculate statistics
       count = len(indices)
       percentage = (count / len(regimes)) * 100
       
       # Return statistics
       mean_return = sum(regime_returns) / count if count > 0 else 0
       regime_vol = math.sqrt(sum(r**2 for r in regime_returns) / count) if count > 0 else 0
       
       # Sharpe ratio (annualized)
       sharpe = (mean_return * 252) / (regime_vol * math.sqrt(252)) if regime_vol > 0 else 0
       
       # ATR statistics
       mean_atr = sum(regime_atr) / count if count > 0 else 0
       
       # Date ranges
       start_date = dates[indices[0]] if indices else None
       end_date = dates[indices[-1]] if indices else None
       
       # Create description based on statistics
       description = regime_description(mean_return, regime_vol, sharpe, mean_atr)
       
       # Add to statistics
       stats.append({
           'regime': regime,
           'count': count,
           'percentage': percentage,
           'mean_return': mean_return,
           'volatility': regime_vol,
           'sharpe': sharpe,
           'mean_atr': mean_atr,
           'start_date': start_date,
           'end_date': end_date,
           'description': description
       })
   
   return stats


def regime_description(mean_return, volatility, sharpe, atr):
   """Generate descriptive label for a market regime"""
   description = []
   
   # Volatility description
   if volatility < 0.005:
       description.append("Low Volatility")
   elif volatility < 0.015:
       description.append("Medium Volatility")
   else:
       description.append("High Volatility")
   
   # Return description
   if mean_return < -0.001:
       description.append("Bearish")
   elif mean_return < 0.001:
       description.append("Sideways")
   else:
       description.append("Bullish")
   
   # Sharpe description
   if abs(sharpe) < 0.5:
       description.append("Inefficient")
   elif sharpe > 1.0:
       description.append("Efficient")
   
   # ATR description
   if atr < 0.01:
       description.append("Low Range")
   elif atr > 0.02:
       description.append("Wide Range")
   
   return ", ".join(description)


def sort_regimes_by_volatility(regimes, regime_stats):
   """Sort regimes by volatility for better interpretation"""
   # Sort regime stats by volatility
   sorted_stats = sorted(regime_stats, key=lambda x: x['volatility'])
   
   # Create mapping from old regime to new regime
   regime_mapping = {stat['regime']: i for i, stat in enumerate(sorted_stats)}
   
   # Map regimes to new indices
   sorted_regimes = [regime_mapping[r] for r in regimes]
   
   return sorted_regimes, regime_mapping


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


def save_results(results, output_dir, ticker='SPY'):
   """Save analysis results to CSV files"""
   # Create output directory if it doesn't exist
   if not os.path.exists(output_dir):
       os.makedirs(output_dir)
   
   # Save regime assignments
   regime_file = os.path.join(output_dir, f"{ticker}_regimes.csv")
   with open(regime_file, 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(['Date', 'Close', 'Return', 'Volatility', 'ATR', 'Regime', 'Regime1_Prob', 'Regime2_Prob', 'Regime3_Prob'])
       
       for i in range(len(results['dates'])):
           # Skip first row (no return calculated)
           if i == 0:
               continue
               
           row = [
               results['dates'][i],
               results['ohlcv']['close'][i],
               results['features']['log_returns'][i-1],  # Offset because returns start at index 1
               results['features']['volatility'][i-1],
               results['features']['atr'][i],
               results['regimes'][i-1] + 1,  # 1-indexed for readability
           ]
           
           # Add regime probabilities
           for p in results['regime_probs'][i-1]:
               row.append(f"{p:.4f}")
               
           writer.writerow(row)
   
   # Save regime statistics
   stats_file = os.path.join(output_dir, f"{ticker}_regime_stats.csv")
   with open(stats_file, 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(['Regime', 'Count', 'Percentage', 'Mean_Return', 'Volatility', 
                        'Sharpe', 'Mean_ATR', 'Start_Date', 'End_Date', 'Description'])
       
       for stat in sorted(results['regime_stats'], key=lambda x: x['regime']):
           writer.writerow([
               stat['regime'] + 1,  # 1-indexed for readability
               stat['count'],
               f"{stat['percentage']:.2f}",
               f"{stat['mean_return']*100:.4f}",
               f"{stat['volatility']*100:.4f}",
               f"{stat['sharpe']:.4f}",
               f"{stat['mean_atr']:.6f}",
               stat['start_date'],
               stat['end_date'],
               stat['description']
           ])
   
   # Save HMM model parameters
   model_file = os.path.join(output_dir, f"{ticker}_hmm_params.csv")
   with open(model_file, 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(['Parameter', 'Value'])
       
       # Save number of regimes
       writer.writerow(['n_regimes', results['hmm_model'].n_regimes])
       
       # Save mean parameters
       for i, mu in enumerate(results['hmm_model'].mu):
           writer.writerow([f'mu_regime_{i+1}', f"{mu:.6f}"])
       
       # Save volatility parameters
       for i, sigma in enumerate(results['hmm_model'].sigma):
           writer.writerow([f'sigma_regime_{i+1}', f"{sigma:.6f}"])
       
       # Save transition matrix
       writer.writerow(['transition_matrix', ''])
       for i, row in enumerate(results['hmm_model'].trans_mat):
           writer.writerow([f'trans_from_regime_{i+1}', ','.join(f"{p:.4f}" for p in row)])
       
       # Save initial probabilities
       writer.writerow(['initial_probs', ','.join(f"{p:.4f}" for p in results['hmm_model'].init_probs)])
   
   print(f"Results saved to {output_dir}")
# Add a function to analyze all regimes across multiple stocks
def analyze_cross_asset_regimes(output_dir):
    """Analyze regime correlations across multiple assets"""
    # This function can be called after running the main analysis
    summary_file = os.path.join(output_dir, "regime_analysis_summary.csv")
    
    if not os.path.exists(summary_file):
        print("No summary file found. Run the main analysis first.")
        return
    
    # Load summary data
    tickers = []
    current_regimes = {}
    
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row['Ticker']
            try:
                regime = int(row['Current_Regime'])
                tickers.append(ticker)
                current_regimes[ticker] = regime
            except:
                # Skip rows with errors
                continue
    
    # Group tickers by regime
    regime_groups = {}
    for ticker, regime in current_regimes.items():
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(ticker)
    
    # Print cross-asset regime analysis
    print("\n=== CROSS-ASSET REGIME ANALYSIS ===")
    print(f"Total assets analyzed: {len(tickers)}")
    
    for regime, ticker_list in sorted(regime_groups.items()):
        percentage = (len(ticker_list) / len(tickers)) * 100
        print(f"Regime {regime}: {len(ticker_list)} assets ({percentage:.1f}%)")
        print(f"  Assets: {', '.join(ticker_list[:10])}" + 
              (f" and {len(ticker_list)-10} more..." if len(ticker_list) > 10 else ""))
    
    # Save cross-asset analysis
    cross_asset_file = os.path.join(output_dir, "cross_asset_regimes.csv")
    with open(cross_asset_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Regime', 'Asset_Count', 'Percentage', 'Assets'])
        
        for regime, ticker_list in sorted(regime_groups.items()):
            percentage = (len(ticker_list) / len(tickers)) * 100
            writer.writerow([
                regime,
                len(ticker_list),
                f"{percentage:.1f}%",
                ', '.join(ticker_list)
            ])
    
    print(f"Cross-asset analysis saved to {cross_asset_file}")

# Update the main function to handle multiple files
def main():
    """Main function to run Bayesian regime detection analysis on multiple stock files"""
    # Set parameters
    data_dir = r"C:\Users\cinco\Desktop\Cinco-Quant\00_raw_data\5.15"  # Your data directory
    output_dir = r"C:\Users\cinco\Desktop\Cinco-Quant\regime_analysis"  # Output directory
    n_regimes = 3  # Number of regimes to identify
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Feature configuration
    feature_config = {
        'volatility_window': 5,
        'atr_window': 14,
        'bb_window': 20,
        'momentum_short': 10,
        'momentum_long': 20,
        'rsi_window': 14
    }
    
    # Feature weights (which features to emphasize)
    feature_weights = {
        'log_returns': 0.15,
        'volatility': 0.30,
        'atr': 0.20,
        'bb_width': 0.15,
        'momentum': 0.10,
        'rsi': 0.10
    }
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*_1daily_*.csv"))
    
    # Create summary file
    summary_file = os.path.join(output_dir, "regime_analysis_summary.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ticker', 'Current_Regime', 'Regime_Description', 
                         'Primary_Strategy', 'Alternative_Strategy', 
                         'Position_Sizing', 'Wing_Width', 'Expiration',
                         'Confidence', 'Transition_Risk'])
    
    # Process each file
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    successful = 0
    failed = 0
    
    for file_path in csv_files:
        # Extract ticker from filename
        file_name = os.path.basename(file_path)
        # Extract ticker from naming convention like SPY_10year_1daily_2010-01-01_to_2025-05-15.csv
        ticker_match = re.match(r'([A-Z]+)_', file_name)
        if ticker_match:
            ticker = ticker_match.group(1)
        else:
            ticker = file_name.split('_')[0]  # Fallback
        
        print(f"\nProcessing {ticker} from {file_name}...")
        
        try:
            # Analyze market regimes
            results = analyze_market_regimes(
                file_path, 
                n_regimes=n_regimes, 
                feature_config=feature_config,
                feature_weights=feature_weights
            )
            
            # Save individual results
            ticker_output_dir = os.path.join(output_dir, ticker)
            if not os.path.exists(ticker_output_dir):
                os.makedirs(ticker_output_dir)
            
            save_results(results, ticker_output_dir, ticker=ticker)
            
            # Generate current regime forecast and strategy recommendation
            current_regime = results['regimes'][-1]
            forecasts = forecast_next_regime(results['hmm_model'], current_regime, horizon=5)
            
            recommendations = recommend_option_strategy(
                regime_stats=results['regime_stats'],
                current_regime=current_regime,
                forecasts=forecasts
            )
            
            # Print recommendations
            print(f"\n=== {ticker} OPTION STRATEGY RECOMMENDATIONS ===")
            print(f"Current Regime: {recommendations['current_regime']}")
            print(f"Primary Strategy: {recommendations['primary_strategy']}")
            print(f"Alternative Strategy: {recommendations['alternative_strategy']}")
            print(f"Position Sizing: {recommendations['position_sizing']}")
            print(f"Wing Width: {recommendations['wing_width']}")
            print(f"Recommended Expiration: {recommendations['expiration']}")
            print(f"Confidence Level: {recommendations['confidence']}")
            print(f"Regime Transition Risk: {recommendations['transition_risk']['description']} ({recommendations['transition_risk']['probability']:.2f})")
            
            # Add to summary file
            with open(summary_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ticker,
                    current_regime + 1,  # 1-indexed for readability
                    recommendations['current_regime'],
                    recommendations['primary_strategy'],
                    recommendations['alternative_strategy'],
                    recommendations['position_sizing'],
                    recommendations['wing_width'],
                    recommendations['expiration'],
                    recommendations['confidence'],
                    f"{recommendations['transition_risk']['description']} ({recommendations['transition_risk']['probability']:.2f})"
                ])
            
            successful += 1
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            failed += 1
            
            # Log error to summary
            with open(summary_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ticker, 'ERROR', str(e), '', '', '', '', '', '', ''])
    
    print(f"\nAnalysis complete. Successfully analyzed {successful} stocks. Failed: {failed}")
    print(f"Results saved to {output_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
   main()

