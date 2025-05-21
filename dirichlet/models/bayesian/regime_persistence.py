import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

class BayesianRegimePersistence:
    def __init__(self, n_regimes=3, features=None):
        """Simple version that doesn't require PyMC"""
        self.n_regimes = n_regimes
        self.feature_names = features if features else []
        self.gmm = None
        self.models = {}
        
    def classify_regimes(self, returns, window=20):
        """Classify regimes using GMM instead of PyMC"""
        # Calculate realized volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna().values.reshape(-1, 1)
        
        # Use Gaussian Mixture Model
        self.gmm = GaussianMixture(
            n_components=self.n_regimes, 
            random_state=42,
            covariance_type='full'
        )
        labels = self.gmm.fit_predict(rolling_vol)
        
        # Map to original time series
        regime_labels = np.zeros(len(returns))
        regime_labels[window-1:window-1+len(labels)] = labels
        
        # Order regimes by volatility (low to high)
        means = self.gmm.means_.flatten()
        order = np.argsort(means)
        mapping = {old: new for new, old in enumerate(order)}
        
        # Remap labels
        for i in range(len(regime_labels)):
            if regime_labels[i] in mapping:
                regime_labels[i] = mapping[regime_labels[i]]
                
        return regime_labels.astype(int)
    
    def fit_persistence_model(self, regimes, features_df=None, lookback=1):
        """Fit logistic regression for persistence probabilities"""
        # Prepare target variable: did the regime stay the same?
        y = np.zeros(len(regimes) - lookback)
        for i in range(len(y)):
            y[i] = 1 if regimes[i + lookback] == regimes[i] else 0
        
        # Prepare features
        X = np.zeros((len(y), lookback + len(self.feature_names)))
        
        # Add regime history
        for i in range(len(y)):
            X[i, :lookback] = regimes[i:i+lookback]
        
        # Add other features if provided
        if features_df is not None and len(self.feature_names) > 0:
            for j, feat in enumerate(self.feature_names):
                X[:, lookback + j] = features_df[feat].values[:len(y)]
        
        # Fit separate model for each regime
        for r in range(self.n_regimes):
            mask = (X[:, 0] == r)
            if sum(mask) < 10:  # Skip if too few examples
                continue
                
            model = LogisticRegression(random_state=42)
            model.fit(X[mask], y[mask])
            self.models[r] = model
        
        return self
    
    def predict_persistence_prob(self, current_regime, recent_regimes=None, features=None):
        """Predict probability of staying in the current regime"""
        if current_regime not in self.models:
            # No model for this regime, use prior probability
            return 0.7  # Prior assumption of persistence
            
        # Prepare features
        X = np.zeros((1, len(self.feature_names) + 1))
        X[0, 0] = current_regime
        
        # Add other features
        if features and len(self.feature_names) > 0:
            for j, feat in enumerate(self.feature_names):
                if feat in features:
                    X[0, 1 + j] = features[feat]
        
        # Predict probability
        return self.models[current_regime].predict_proba(X)[0, 1]
    
    def classify_current_regime(self, recent_returns, window=20):
        """Classify the current market regime"""
        if len(recent_returns) < window:
            raise ValueError(f"Need at least {window} returns to calculate volatility")
            
        # Calculate realized volatility
        vol = np.std(recent_returns[-window:]) * np.sqrt(252)
        vol_array = np.array([[vol]])
        
        # Predict regime probabilities
        probs = self.gmm.predict_proba(vol_array)[0]
        regime = np.argmax(probs)
        
        return regime, probs
    
    def forecast_regime(self, current_regime, horizon=5, features=None, feature_forecast=None):
        """Forecast regime probabilities"""
        # Initialize with current regime
        regime_probs = np.zeros((horizon, self.n_regimes))
        regime_probs[0, current_regime] = 1.0
        
        for t in range(1, horizon):
            # For each possible current regime
            for r in range(self.n_regimes):
                if regime_probs[t-1, r] > 0:
                    # Probability of staying in regime r
                    stay_prob = self.predict_persistence_prob(r)
                    
                    # Add to probability of being in regime r
                    regime_probs[t, r] += regime_probs[t-1, r] * stay_prob
                    
                    # Distribute remaining probability to other regimes
                    transition_prob = 1 - stay_prob
                    for other_r in range(self.n_regimes):
                        if other_r != r:
                            # Simplified: equal distribution among other regimes
                            regime_probs[t, other_r] += regime_probs[t-1, r] * transition_prob / (self.n_regimes - 1)
        
        return regime_probs