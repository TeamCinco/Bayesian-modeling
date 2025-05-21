# transition_analyzer.py
"""
Improved functions for counting regime transitions, Bayesian analysis, and distributions.
"""
import numpy as np
import pandas as pd

def count_transitions_improved(regimes: pd.Series) -> tuple:
    """Count regime transitions robustly. Returns counts and observed regime labels."""
    regimes_clean = regimes.dropna().astype(int) # Ensure integer type after dropna
    if regimes_clean.empty:
        return np.zeros((0, 0)), np.array([], dtype=int)

    unique_regime_labels = sorted(np.unique(regimes_clean))
    if not unique_regime_labels: # Should not happen if not empty, but defensive
        return np.zeros((0, 0)), np.array([], dtype=int)

    # Map potentially sparse labels (e.g., 0, 2) to dense indices (0, 1) for matrix
    label_to_idx_map = {label: i for i, label in enumerate(unique_regime_labels)}
    num_distinct_observed_regimes = len(unique_regime_labels)

    transition_counts = np.zeros((num_distinct_observed_regimes, num_distinct_observed_regimes), dtype=int)
    
    prev_mapped_idx = None
    for raw_label in regimes_clean:
        current_mapped_idx = label_to_idx_map[raw_label]
        if prev_mapped_idx is not None:
            transition_counts[prev_mapped_idx, current_mapped_idx] += 1
        prev_mapped_idx = current_mapped_idx
    
    # Return matrix indexed by dense observed labels, and the original labels themselves
    return transition_counts, np.array(unique_regime_labels, dtype=int)


def bayesian_transition_analysis_improved(transition_counts: np.ndarray,
                                          regime_observations: np.ndarray, # Total observations in each 'from_regime'
                                          prior_type: str = 'jeffrey',
                                          min_prior_strength: float = 0.1,
                                          max_prior_strength: float = 2.0) -> tuple:
    """Performs Bayesian analysis on transition counts with adaptive priors."""
    num_regimes_from_tc = transition_counts.shape[0]
    if num_regimes_from_tc == 0:
        return np.array([]), {}, {}

    # If regime_observations doesn't match tc, it implies tc might be subsetted.
    # The tc passed here should be for the "active" regimes from count_transitions_improved.
    # So, regime_observations should correspond to rows of tc.
    
    mean_probs_matrix = np.zeros_like(transition_counts, dtype=float)
    posterior_samples_dict = {}
    credible_intervals_dict = {}
    rng = np.random.default_rng()

    for from_idx in range(num_regimes_from_tc):
        current_obs_count = regime_observations[from_idx] if from_idx < len(regime_observations) else 0

        if current_obs_count == 0: # No observations from this state
            # Assign uniform probabilities and wide CIs
            uniform_prob = 1.0 / num_regimes_from_tc if num_regimes_from_tc > 0 else 0
            mean_probs_matrix[from_idx, :] = uniform_prob
            # Samples for CI calculation should still be generated to maintain structure
            posterior_samples_dict[from_idx] = rng.dirichlet(np.ones(num_regimes_from_tc) * 1e-9, size=10000) \
                                                if num_regimes_from_tc > 0 else np.array([]) # small alpha if 0 obs
            credible_intervals_dict[from_idx] = {
                'lower': np.full(num_regimes_from_tc, 0.0), 'upper': np.full(num_regimes_from_tc, 1.0),
                'width': np.full(num_regimes_from_tc, 1.0), 'samples_count': 0}
            continue

        # Determine prior alphas based on type
        if prior_type == 'jeffrey': prior_alphas_base = np.ones(num_regimes_from_tc) * 0.5
        elif prior_type == 'uniform': prior_alphas_base = np.ones(num_regimes_from_tc)
        elif prior_type == 'empirical':
            # Overall transition freqs, smoothed
            col_sums = transition_counts.sum(axis=0)
            total_transitions = col_sums.sum()
            if total_transitions > 0:
                empirical_dist = (col_sums + 0.1) / (total_transitions + 0.1 * num_regimes_from_tc) # Smoothed
                prior_alphas_base = empirical_dist 
            else: prior_alphas_base = np.ones(num_regimes_from_tc) * (1.0 / num_regimes_from_tc)
        else: prior_alphas_base = np.ones(num_regimes_from_tc) # Default to uniform type

        # Adaptive prior strength based on number of observations from this state
        # Less data => stronger prior influence (closer to max_prior_strength)
        # More data => weaker prior influence (closer to min_prior_strength)
        adaptive_strength = max(min_prior_strength, 
                                min(max_prior_strength, max_prior_strength * np.exp(-current_obs_count / 50.0)))
        final_prior_alphas = prior_alphas_base * adaptive_strength
        
        posterior_alphas = final_prior_alphas + transition_counts[from_idx]
        # Ensure alphas are positive for Dirichlet
        safe_posterior_alphas = np.maximum(posterior_alphas, 1e-9)
        
        samples = rng.dirichlet(safe_posterior_alphas, size=10000)
        posterior_samples_dict[from_idx] = samples
        mean_probs_matrix[from_idx, :] = safe_posterior_alphas / safe_posterior_alphas.sum()
        
        lower_ci = np.percentile(samples, 2.5, axis=0)
        upper_ci = np.percentile(samples, 97.5, axis=0)
        credible_intervals_dict[from_idx] = {
            'lower': lower_ci, 'upper': upper_ci,
            'width': upper_ci - lower_ci, 'samples_count': current_obs_count}
            
    return mean_probs_matrix, posterior_samples_dict, credible_intervals_dict

def calculate_transition_probability_distributions(posterior_samples: dict, 
                                                   credible_intervals: dict) -> tuple:
    """Calculate PMF-like histograms and CDFs from posterior samples."""
    pmfs_dict = {}
    cdfs_dict = {}

    for from_regime_idx, samples_array in posterior_samples.items():
        if samples_array.ndim != 2 or samples_array.shape[1] == 0: continue # Skip if no 'to_regimes'
        
        num_to_regimes = samples_array.shape[1]
        current_regime_pmfs = {}
        current_regime_cdfs = {}

        obs_count_for_cis = credible_intervals.get(from_regime_idx, {}).get('samples_count', 0)

        for to_regime_idx in range(num_to_regimes):
            prob_samples_for_transition = samples_array[:, to_regime_idx]
            
            hist_vals, bin_edges = np.histogram(prob_samples_for_transition, bins=20, range=(0,1), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            
            sorted_prob_samples = np.sort(prob_samples_for_transition)
            cdf_y_values = np.arange(1, len(sorted_prob_samples) + 1) / len(sorted_prob_samples)
            
            current_regime_pmfs[to_regime_idx] = {
                'bins': bin_centers.tolist(), 'density': hist_vals.tolist(),
                'mean': np.mean(prob_samples_for_transition),
                'median': np.median(prob_samples_for_transition),
                'std_dev': np.std(prob_samples_for_transition)
            }
            current_regime_cdfs[to_regime_idx] = {
                'probabilities': sorted_prob_samples.tolist(), 'cumulative_prob': cdf_y_values.tolist(),
                'samples_count': obs_count_for_cis # Re-affirm using samples_count from CIs
            }
        pmfs_dict[from_regime_idx] = current_regime_pmfs
        cdfs_dict[from_regime_idx] = current_regime_cdfs
        
    return pmfs_dict, cdfs_dict