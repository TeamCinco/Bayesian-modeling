# gmm_regime_identifier.py
"""
Implements Gaussian Mixture Models (GMM) for volatility regime identification.
Includes 1D and 2D GMM identification.
"""
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# Existing identify_volatility_regimes_gmm (1D) - content from previous, no changes
def identify_volatility_regimes_gmm(volatility: pd.Series,
                                    num_regimes_target: int = 3,
                                    random_state: int = 42) -> tuple:
    clean_vol = volatility.dropna()
    min_samples_needed = num_regimes_target * 5
    default_regime_info = {
        'model': None, 'mapping': {},
        'means': [np.nan] * num_regimes_target,
        'variances': [np.nan] * num_regimes_target,
        'weights': [np.nan] * num_regimes_target
    }
    if len(clean_vol) < min_samples_needed or len(clean_vol.unique()) < num_regimes_target:
        regimes = pd.Series(np.nan, index=volatility.index)
        if not clean_vol.empty and num_regimes_target >= 1:
            regimes.loc[clean_vol.index] = 0
            current_regime_info = default_regime_info.copy()
            current_regime_info['mapping'] = {0:0}
            if num_regimes_target > 0:
                current_regime_info['means'][0] = clean_vol.mean()
                current_regime_info['variances'][0] = clean_vol.var(ddof=0) if len(clean_vol) > 0 else 0.0
                current_regime_info['weights'][0] = 1.0
                for i in range(1, num_regimes_target): current_regime_info['weights'][i] = 0.0
            return regimes, current_regime_info
        return regimes, default_regime_info
    X = clean_vol.values.reshape(-1, 1)
    if np.all(X == X[0]):
        regimes = pd.Series(np.nan, index=volatility.index)
        regimes.loc[clean_vol.index] = 0
        current_regime_info = default_regime_info.copy()
        current_regime_info['mapping'] = {0:0}
        if num_regimes_target > 0:
            current_regime_info['means'][0] = X[0][0]
            current_regime_info['variances'][0] = 0.0
            current_regime_info['weights'][0] = 1.0
            for i in range(1, num_regimes_target): current_regime_info['weights'][i] = 0.0
        return regimes, current_regime_info
    gmm = GaussianMixture(n_components=num_regimes_target, covariance_type='full',
                          random_state=random_state, n_init=10, reg_covar=1e-6)
    try:
        gmm.fit(X)
        # ... (rest of the 1D GMM function from previous iteration) ...
        regime_labels_raw = gmm.predict(X)
        gmm_means_flat = gmm.means_.flatten()
        regime_order = np.argsort(gmm_means_flat)
        regime_mapping = {old_label: new_label for new_label, old_label in enumerate(regime_order)}
        regimes_clean = pd.Series(regime_labels_raw, index=clean_vol.index).map(regime_mapping)
        regimes = pd.Series(np.nan, index=volatility.index)
        regimes.loc[regimes_clean.index] = regimes_clean
        actual_fitted_components = gmm.n_components
        final_regime_info = {
            'model': gmm, 'mapping': regime_mapping,
            'means': [gmm.means_[regime_order[i]][0] if i < actual_fitted_components else np.nan
                      for i in range(num_regimes_target)],
            'variances': [gmm.covariances_[regime_order[i]][0][0] if i < actual_fitted_components else np.nan
                          for i in range(num_regimes_target)],
            'weights': [gmm.weights_[regime_order[i]] if i < actual_fitted_components else 0.0
                        for i in range(num_regimes_target)]
        }
        return regimes, final_regime_info

    except ValueError as e: # GMM fitting error
        print(f"GMM fitting error fallback for 1D ({clean_vol.name if hasattr(clean_vol, 'name') else ''}): {e}.")
        regimes = pd.Series(np.nan, index=volatility.index); regimes.loc[clean_vol.index] = 0
        current_regime_info = default_regime_info.copy(); current_regime_info['mapping'] = {0:0}
        if num_regimes_target > 0:
            current_regime_info['means'][0] = clean_vol.mean()
            current_regime_info['variances'][0] = clean_vol.var(ddof=0) if len(clean_vol)>0 else np.nan
            current_regime_info['weights'][0] = 1.0
        return regimes, current_regime_info


# New 2D GMM regime identifier
def identify_volatility_regimes_gmm_improved(current_volatility: pd.Series,
                                             forecasted_volatility: pd.Series,
                                             num_regimes_target: int = 3,
                                             random_state: int = 42) -> tuple:
    """Use both current and forecasted volatility for regime detection (2D GMM)."""
    combined_data = pd.DataFrame({
        'current_vol': current_volatility,
        'forecasted_vol': forecasted_volatility
    }).dropna()

    default_regime_info = {
        'model': None, 'mapping': {},
        'means': [np.array([np.nan, np.nan])] * num_regimes_target, # Store 2D means
        'covariances': [np.full((2,2), np.nan)] * num_regimes_target,
        'weights': [np.nan] * num_regimes_target
    }

    if len(combined_data) < num_regimes_target * 10: # Need more samples for 2D
        regimes_out = pd.Series(np.nan, index=current_volatility.index)
        # Fallback: if not enough for 2D, could try 1D on current_volatility
        # print(f"Not enough data for 2D GMM ({len(combined_data)} points).")
        return regimes_out, default_regime_info

    X = combined_data.values
    if X.shape[1] != 2:
        # print("Combined data for 2D GMM does not have 2 dimensions.")
        return pd.Series(np.nan, index=current_volatility.index), default_regime_info

    try:
        gmm = GaussianMixture(n_components=num_regimes_target, covariance_type='full',
                              random_state=random_state, n_init=10, reg_covar=1e-6)
        gmm.fit(X)
        
        regime_labels_raw = gmm.predict(X)
        
        # Sort regimes by the mean of the 'current_vol' dimension (index 0 of means)
        # GMM means will be of shape (n_components, n_features=2)
        component_means_current_vol = gmm.means_[:, 0]
        regime_order = np.argsort(component_means_current_vol)
        
        regime_mapping = {old_label: new_label for new_label, old_label in enumerate(regime_order)}
        regimes_clean = pd.Series(regime_labels_raw, index=combined_data.index).map(regime_mapping)
        
        regimes_out = pd.Series(np.nan, index=current_volatility.index)
        regimes_out.loc[regimes_clean.index] = regimes_clean
        
        actual_fitted_components = gmm.n_components
        final_regime_info = {
            'model': gmm, 'mapping': regime_mapping,
            'means': [gmm.means_[regime_order[i]] if i < actual_fitted_components else np.array([np.nan,np.nan])
                      for i in range(num_regimes_target)],
            'covariances': [gmm.covariances_[regime_order[i]] if i < actual_fitted_components else np.full((2,2),np.nan)
                            for i in range(num_regimes_target)],
            'weights': [gmm.weights_[regime_order[i]] if i < actual_fitted_components else 0.0
                        for i in range(num_regimes_target)]
        }
        return regimes_out, final_regime_info
    except Exception as e:
        # print(f"Error in 2D GMM fitting: {e}")
        return pd.Series(np.nan, index=current_volatility.index), default_regime_info