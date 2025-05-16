import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math
import scipy.stats as stats
from datetime import datetime
from sklearn.mixture import GaussianMixture
from scipy.stats import norm # Ensure norm is imported

# Set the path to your data
DATA_PATH = "/Users/jazzhashzzz/Desktop/Cinco-Quant/00_raw_data/5.15"

def load_stock_data(folder_path):
    """Load all CSV files from a folder into a dictionary of dataframes."""
    all_data = {}
    for file_path in glob.glob(os.path.join(folder_path, "*.csv")):
        symbol = os.path.basename(file_path).split('.')[0]
        try:
            print(f"Loading {symbol}...")
            df = pd.read_csv(file_path)
            
            date_cols = [col for col in df.columns if any(date_str in col.lower() 
                                                       for date_str in ['date', 'time'])]
            
            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                except Exception as e:
                    print(f"Could not parse dates in {symbol} ({date_col}): {e}, skipping date conversion")
            
            if 'close' in df.columns:
                all_data[symbol] = df
                print(f"Successfully loaded {symbol} with {len(df)} rows")
            else:
                close_cols = [col for col in df.columns if 'close' in col.lower()]
                if close_cols:
                    df.rename(columns={close_cols[0]: 'close'}, inplace=True)
                    all_data[symbol] = df
                    print(f"Renamed column {close_cols[0]} to 'close' for {symbol}")
                else:
                    print(f"Skipping {symbol}: no 'close' column found")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_data

def calculate_volatility(df, window=20):
    """Calculate rolling volatility from price data."""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

def identify_volatility_regimes(volatility, num_regimes=3): # Original quantile-based, not used by GMM flow but kept for completeness
    """Identify volatility regimes using simple quantile thresholds."""
    clean_vol = volatility.dropna()
    if clean_vol.empty:
        print("Warning: No valid volatility data for regime identification")
        return pd.Series(index=volatility.index)
    if num_regimes == 2:
        threshold = np.median(clean_vol)
        regimes = pd.Series(0, index=volatility.index)
        regimes[volatility > threshold] = 1
    else:
        thresholds = np.linspace(0, 100, num_regimes+1)[1:-1]
        quantiles = np.percentile(clean_vol, thresholds)
        regimes = pd.Series(0, index=volatility.index)
        for i in range(1, num_regimes):
            regimes[volatility > quantiles[i-1]] = i
    return regimes

def count_transitions(regimes):
    """Count transitions between volatility regimes."""
    regimes_clean = regimes.dropna()
    if regimes_clean.empty:
        print("Warning: No valid regime data for transition counting")
        return np.zeros((1, 1)) # Return a shape that won't break num_regimes calc later
    
    unique_regimes_from_data = sorted(regimes_clean.unique().astype(int))
    if not unique_regimes_from_data: # Should be caught by regimes_clean.empty but as safeguard
        return np.zeros((1,1))

    # Determine num_regimes from data or expected if GMM might not find all
    # This assumes regimes are 0-indexed up to num_expected_regimes - 1
    # If identify_volatility_regimes_gmm guarantees num_regimes components, this is safer:
    # num_regimes = len(gmm_model.means_) or similar if passed around
    # For now, let's assume max regime label indicates number of regimes
    if not unique_regimes_from_data: # Should be caught by regimes_clean.empty but as safeguard
        return np.zeros((1,1))
    num_regimes_found = max(unique_regimes_from_data) + 1 if unique_regimes_from_data else 0


    transition_counts = np.zeros((num_regimes_found, num_regimes_found))
    
    prev_regime = None
    for regime_val in regimes_clean:
        regime = int(regime_val)
        if prev_regime is not None:
            if 0 <= prev_regime < num_regimes_found and 0 <= regime < num_regimes_found:
                 transition_counts[prev_regime, regime] += 1
            else:
                print(f"Warning: Regime out of bounds. Prev: {prev_regime}, Curr: {regime}, Max_idx: {num_regimes_found-1}")
        prev_regime = regime
        
    return transition_counts

def filter_data_by_date(data, start_date='2021-01-01', end_date=None):
    """Filter data to only include dates within the specified range."""
    if not isinstance(data.index, pd.DatetimeIndex):
        print("Warning: Data index is not a DatetimeIndex, skipping date filtering")
        return data
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize() # Normalize for consistent comparison
    
    # Ensure correct slicing: data.index should be sorted
    filtered_data = data.loc[(data.index >= start) & (data.index <= end)].copy()
    if not filtered_data.empty:
        print(f"Filtered data from {filtered_data.index.min()} to {filtered_data.index.max()}")
    else:
        print(f"No data found in range {start} to {end}")
    print(f"Reduced from {len(data)} to {len(filtered_data)} data points for range {start_date} to {end_date}")
    
    return filtered_data

def bayesian_transition_analysis(transition_counts, prior_strength=1.0):
    """Perform Bayesian analysis of regime transitions using Dirichlet-Multinomial model."""
    num_regimes = transition_counts.shape[0]
    if num_regimes == 0 or (num_regimes == 1 and transition_counts.shape[1] == 1 and transition_counts[0,0] == 0): # Handle edge case from count_transitions
        print("Warning: No regimes or valid transitions identified for Bayesian analysis")
        return np.array([]), {}, {}
    
    posterior_samples = {}
    mean_probs = np.zeros_like(transition_counts, dtype=float)
    
    for from_regime in range(num_regimes):
        prior_alphas = np.ones(num_regimes) * prior_strength
        posterior_alphas = prior_alphas + transition_counts[from_regime]
        
        if posterior_alphas.sum() == 0: # Avoid error with dirichlet if all counts are zero for a row
            samples = np.zeros((10000, num_regimes)) # Or handle as appropriate
            mean_probs[from_regime] = np.zeros(num_regimes)
        else:
            samples = np.random.default_rng().dirichlet(posterior_alphas, size=10000)
            mean_probs[from_regime] = posterior_alphas / posterior_alphas.sum()
        posterior_samples[from_regime] = samples
            
    return mean_probs, posterior_samples

def calculate_probability_intervals(samples, confidence=0.95):
    """Calculate credible intervals for transition probabilities."""
    lower_bound = (1 - confidence) / 2
    upper_bound = 1 - lower_bound
    lower_probs = {}
    upper_probs = {}
    
    for from_regime, regime_samples in samples.items():
        lower_probs[from_regime] = np.quantile(regime_samples, lower_bound, axis=0)
        upper_probs[from_regime] = np.quantile(regime_samples, upper_bound, axis=0)
        
    return lower_probs, upper_probs

def identify_volatility_regimes_gmm(volatility, num_regimes=3, random_state=42):
    """Identify volatility regimes using Gaussian Mixture Models."""
    clean_vol = volatility.dropna()
    if clean_vol.empty:
        print("Warning: No valid volatility data for GMM regime identification")
        # Return structure consistent with successful run but with NaN regimes
        regimes = pd.Series(np.nan, index=volatility.index)
        regime_info = {
            'model': None, 'mapping': {}, 'means': [np.nan]*num_regimes,
            'variances': [np.nan]*num_regimes, 'weights': [np.nan]*num_regimes
        }
        return regimes, regime_info
    
    X = clean_vol.values.reshape(-1, 1)
    
    gmm = GaussianMixture(
        n_components=num_regimes, covariance_type='full',
        random_state=random_state, n_init=10
    )
    gmm.fit(X)
    
    regime_labels = gmm.predict(X)
    regimes_clean = pd.Series(regime_labels, index=clean_vol.index)
    
    means = [gmm.means_[i][0] for i in range(num_regimes)]
    regime_order = np.argsort(means)
    regime_mapping = {old_label: new_label for new_label, old_label in enumerate(regime_order)}
    
    regimes_clean = regimes_clean.map(regime_mapping)
    
    regimes = pd.Series(np.nan, index=volatility.index)
    regimes.loc[regimes_clean.index] = regimes_clean
    
    regime_info = {
        'model': gmm, 'mapping': regime_mapping,
        'means': [gmm.means_[regime_order[i]][0] for i in range(num_regimes)],
        'variances': [gmm.covariances_[regime_order[i]][0][0] for i in range(num_regimes)],
        'weights': [gmm.weights_[regime_order[i]] for i in range(num_regimes)]
    }
    return regimes, regime_info

def plot_gmm_regimes(volatility, regimes, regime_info, title=None, ax=None, 
                     start_date_dt=None, end_date_dt=None):
    """Plot the volatility data with GMM-identified regimes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    clean_vol = volatility.dropna()
    
    volatility.plot(ax=ax, color='black', alpha=0.3, label='Volatility (underlying)')
    
    num_regimes = len(regime_info['means'])
    colors = plt.cm.tab10(np.linspace(0, 1, num_regimes))
    
    if not clean_vol.empty:
        for i in range(num_regimes):
            # Ensure mask is aligned with clean_vol.index before applying
            mask_full_idx = (regimes == i) # Mask on original volatility index (with NaNs)
            aligned_mask = mask_full_idx.reindex(clean_vol.index, fill_value=False)

            ax.scatter(clean_vol.index[aligned_mask], 
                       clean_vol[aligned_mask], 
                       color=colors[i], label=f'Regime {i} Data', alpha=0.7) # Label for clarity, will be overridden
    else:
        print("No clean volatility data to scatter plot.")

    # Annotations for regime parameters
    # Determine x-position for annotations on the time axis
    if start_date_dt:
        annotation_time_x = start_date_dt + pd.Timedelta(days=3) # Small offset from the start
        if not clean_vol.empty and annotation_time_x < clean_vol.index.min():
            annotation_time_x = clean_vol.index.min() + pd.Timedelta(days=3)
        elif clean_vol.empty: # If no data, place at start_date_dt
             annotation_time_x = start_date_dt
    elif not clean_vol.empty: # Fallback if start_date_dt is not provided but data exists
        annotation_time_x = clean_vol.index.min() + pd.Timedelta(days=3)
    else: # Absolute fallback
        annotation_time_x = pd.Timestamp('now') - pd.Timedelta(days=365) # Default if no info

    # Y-axis limits for annotation placement
    y_min_ax, y_max_ax = ax.get_ylim()

    for i in range(num_regimes):
        mean = regime_info['means'][i]
        variance = regime_info['variances'][i]
        if pd.isna(mean) or pd.isna(variance): # Skip if GMM failed for this regime
            continue

        # Place text annotations at the `mean` volatility level, at `annotation_time_x`
        ax.text(annotation_time_x, mean,
                f'μ={mean:.3f}\nσ={np.sqrt(variance):.3f}',
                horizontalalignment='left', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=colors[i], boxstyle='round,pad=0.3'),
                fontsize=8, color=colors[i])
        
        # Draw a dashed line from y=0 (or axis bottom) to the mean volatility at annotation_time_x
        ax.plot([annotation_time_x, annotation_time_x], [y_min_ax, mean],
                color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)

    # Custom legend
    regime_descriptions = {
        0: "Low Volatility",
        1: "Medium Volatility" if num_regimes > 2 else "High Volatility",
        2: "High Volatility" if num_regimes > 2 else None # Only relevant if num_regimes is 3+
    }
    legend_elements = []
    for i in range(num_regimes):
        desc = regime_descriptions.get(i)
        label_text = f'Regime {i}: {desc}' if desc else f'Regime {i}'
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], markersize=8,
                                        label=label_text))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Twin axis for PDF (removed actual PDF plotting to fix 1970s axis)
    ax_twin = ax.twinx()
    ax_twin.set_ylabel('Probability Density (Distributions Not Plotted)', fontsize=10)
    ax_twin.set_yticks([]) # Hide y-ticks as no PDF is plotted here

    # Set x-axis limits based on provided dates and data
    if not clean_vol.empty:
        data_min_date = clean_vol.index.min()
        data_max_date = clean_vol.index.max()
        
        plot_start = start_date_dt if start_date_dt else data_min_date
        plot_end = end_date_dt if end_date_dt else data_max_date

        # Ensure plot_start and plot_end are within actual data range if specified
        plot_start = max(plot_start, data_min_date)
        plot_end = min(plot_end, data_max_date)

        if plot_start < plot_end:
            ax.set_xlim(plot_start, plot_end)
        else: # Fallback if range is invalid
            ax.set_xlim(data_min_date, data_max_date)
    elif start_date_dt and end_date_dt and start_date_dt < end_date_dt: # No data but valid range given
        ax.set_xlim(start_date_dt, end_date_dt)

    if title:
        ax.set_title(title, fontsize=14)
    
    ax.set_xlabel("datetime") # Explicitly label x-axis
    return ax, ax_twin

def analyze_stock_with_gmm(symbol, data, window=20, num_regimes=3, prior_strength=1.0, 
                          start_date='2021-01-01', end_date='2025-05-14'):
    print(f"\nAnalyzing {symbol} with GMM clustering for period {start_date} to {end_date}...")
    
    original_data_len = len(data)
    if isinstance(data.index, pd.DatetimeIndex):
        print(f"Original data range: {data.index.min()} to {data.index.max()} ({original_data_len} rows)")
        data = filter_data_by_date(data, start_date, end_date)
        if data.empty:
            print(f"No data remaining for {symbol} after filtering for dates {start_date} to {end_date}.")
            return None
    else:
        print("Data index is not datetime format, cannot filter by date. Using all data.")
    
    volatility = calculate_volatility(data, window)
    
    if volatility.dropna().empty:
        print(f"Not enough data for {symbol} to calculate volatility after filtering and windowing.")
        return None
        
    regimes, regime_info = identify_volatility_regimes_gmm(volatility, num_regimes)
    
    if all(pd.isna(val) for val in regime_info['means']): # Check if GMM failed
        print(f"GMM failed to identify regimes for {symbol}.")
        return None

    transition_counts = count_transitions(regimes)
    
    # Ensure transition_counts has expected shape based on num_regimes identified by GMM
    # GMM might find fewer components than requested if data doesn't support them.
    # The regime_info['means'] length is a good indicator of actual regimes found and ordered.
    actual_num_regimes_found = len([m for m in regime_info['means'] if not pd.isna(m)])
    
    if transition_counts.shape[0] == 0 or transition_counts.shape[0] != actual_num_regimes_found :
        print(f"Transition count matrix shape mismatch or empty for {symbol}. Expected {actual_num_regimes_found}, got {transition_counts.shape}. Adjusting...")
        # This might happen if some regimes are never visited after GMM sorting, or if count_transitions has issues.
        # Re-initialize transition_counts if it's problematic.
        # For now, we'll proceed, but Bayesian analysis might be affected.
        if actual_num_regimes_found > 0 and (transition_counts.shape[0] != actual_num_regimes_found):
            # If GMM found regimes, but transitions are mismatched, this is an issue.
            # This part needs careful handling based on how count_transitions reports num_regimes.
            # Simplest fix: if count_transitions is based on max(regime_label), ensure it aligns.
            # For now, if GMM says 3 regimes, but transitions only show 2, Bayesian analysis might use 2.
            pass # Let bayesian_transition_analysis handle its input

    mean_probs, posterior_samples = bayesian_transition_analysis(transition_counts, prior_strength)
    
    if not posterior_samples: # Check if posterior_samples is empty
        print(f"Not enough regime transitions or Bayesian analysis failed for {symbol}")
        return None
    
    lower_probs, upper_probs = calculate_probability_intervals(posterior_samples)
    
    current_regime = None
    if not regimes.dropna().empty:
        current_regime = int(regimes.dropna().iloc[-1])
    else:
        print(f"No valid current regime for {symbol}")
        return None
    
    if current_regime not in posterior_samples: # Ensure current_regime is a valid key
        print(f"Current regime {current_regime} not found in posterior samples keys for {symbol}. Keys: {posterior_samples.keys()}")
        # This can happen if current_regime was not a 'from_regime' with transitions.
        # Fallback or error handling needed here. For now, we'll let it potentially error in results prep.
        # Or, more safely:
        return None


    results = {
        'symbol': symbol, 'volatility': volatility, 'regimes': regimes,
        'regime_info': regime_info, 'transition_counts': transition_counts,
        'transition_probabilities': mean_probs, 'posterior_samples': posterior_samples,
        'lower_probabilities': lower_probs, 'upper_probabilities': upper_probs,
        'current_regime': current_regime
    }
    
    print(f"Current volatility regime: {current_regime}")
    print("\nRegime characteristics:")
    for i in range(actual_num_regimes_found): # Use actual_num_regimes_found
        mean = regime_info['means'][i]
        std_dev = np.sqrt(regime_info['variances'][i])
        weight = regime_info['weights'][i]
        print(f"  Regime {i}: Mean Volatility = {mean:.3f}, Std Dev = {std_dev:.3f}, Weight = {weight:.3f}")
    
    print("\nTransition probability matrix:")
    # Ensure mean_probs DataFrame uses actual_num_regimes_found for labels
    df_probs_idx_cols = [f'Regime {i}' for i in range(mean_probs.shape[0])] # Use actual shape of mean_probs
    df_probs = pd.DataFrame(mean_probs, index=df_probs_idx_cols, columns=df_probs_idx_cols)
    print(df_probs)
    
    print(f"\nProbabilities of transitioning from current regime {current_regime}:")
    if current_regime < mean_probs.shape[0]: # Check bounds
        current_probs_row = mean_probs[current_regime]
        current_lower_row = lower_probs[current_regime]
        current_upper_row = upper_probs[current_regime]
        for to_regime in range(mean_probs.shape[1]): # Iterate over columns of mean_probs
            print(f"  To Regime {to_regime}: {current_probs_row[to_regime]:.3f} " +
                  f"(95% CI: {current_lower_row[to_regime]:.3f} - {current_upper_row[to_regime]:.3f})")
    else:
        print(f"  Current regime {current_regime} is out of bounds for the transition probability matrix.")

    return results

def visualize_gmm_results(results, save_dir=None, start_date_str=None, end_date_str=None):
    """Create enhanced visualization of the GMM-based volatility regimes and transition probabilities."""
    symbol = results['symbol']
    volatility = results['volatility']
    regimes = results['regimes']
    regime_info = results['regime_info']
    mean_probs = results['transition_probabilities']
    current_regime = results['current_regime']
    
    if current_regime not in results['lower_probabilities'] or current_regime not in results['upper_probabilities']:
        print(f"Cannot visualize: Current regime {current_regime} missing from probability intervals for {symbol}.")
        return

    lower_probs_current = results['lower_probabilities'][current_regime]
    upper_probs_current = results['upper_probabilities'][current_regime]
    
    actual_num_regimes = mean_probs.shape[0] # Based on the transition matrix
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2.5, 2, 1.5]})
    fig.suptitle(f'{symbol} Volatility Regime Analysis ({start_date_str} to {end_date_str})', fontsize=16, y=0.99)
    
    # Convert string dates to Timestamps for plotting functions
    start_date_dt = pd.Timestamp(start_date_str) if start_date_str else None
    end_date_dt = pd.Timestamp(end_date_str) if end_date_str else None

    title_plot1 = f'{symbol} - Volatility Regimes with GMM Clustering'
    ax1, _ = plot_gmm_regimes(volatility, regimes, regime_info, title_plot1, ax=axes[0],
                              start_date_dt=start_date_dt, end_date_dt=end_date_dt)
    
    ax2 = axes[1]
    if not regimes.dropna().empty:
        regimes.plot(ax=ax2, drawstyle='steps-post', linewidth=2)
    colors = plt.cm.tab10(np.linspace(0, 1, actual_num_regimes)) # Use actual_num_regimes
    
    for i in range(actual_num_regimes): # Use actual_num_regimes
        mask_full_idx = (regimes == i)
        if mask_full_idx.any():
            # Find continuous periods for axvspan
            # This part can be tricky if regimes series has NaNs not aligned with GMM output
            # Let's try a robust way to find spans on the non-NaN parts of regimes
            valid_regimes = regimes.dropna()
            mask_valid = (valid_regimes == i)
            if mask_valid.any():
                # Convert boolean mask to integer (0 or 1) to find changes
                int_mask = mask_valid.astype(int)
                diff_mask = int_mask.diff()
                
                starts = valid_regimes.index[diff_mask == 1]
                ends = valid_regimes.index[diff_mask == -1]

                # Handle cases where period starts at beginning or ends at end
                if mask_valid.iloc[0]:
                    starts = pd.Index([valid_regimes.index[0]]).append(starts)
                if mask_valid.iloc[-1]:
                    ends = ends.append(pd.Index([valid_regimes.index[-1]]))
                
                for start, end in zip(starts, ends):
                    ax2.axvspan(start, end, alpha=0.2, color=colors[i])
    
    ax2.set_yticks(range(actual_num_regimes))
    regime_descriptions_short = {
        0: "Low Vol", 1: "Med Vol" if actual_num_regimes > 2 else "High Vol",
        2: "High Vol" if actual_num_regimes > 2 else None
    }
    ax2.set_yticklabels([regime_descriptions_short.get(i, f"Reg {i}") for i in range(actual_num_regimes)])
    ax2.set_title(f'{symbol} - Volatility Regime Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)

    if not regimes.dropna().empty:
        data_min_date_ax2 = regimes.dropna().index.min()
        data_max_date_ax2 = regimes.dropna().index.max()
        plot_start_ax2 = start_date_dt if start_date_dt else data_min_date_ax2
        plot_end_ax2 = end_date_dt if end_date_dt else data_max_date_ax2
        plot_start_ax2 = max(plot_start_ax2, data_min_date_ax2)
        plot_end_ax2 = min(plot_end_ax2, data_max_date_ax2)
        if plot_start_ax2 < plot_end_ax2:
            ax2.set_xlim(plot_start_ax2, plot_end_ax2)
    elif start_date_dt and end_date_dt and start_date_dt < end_date_dt:
        ax2.set_xlim(start_date_dt, end_date_dt)
    ax2.set_xlabel("datetime")


    ax2_twin = ax2.twinx()
    if not volatility.empty:
        volatility.plot(ax=ax2_twin, color='gray', alpha=0.5)
    ax2_twin.set_ylabel('Volatility')
    
    ax3 = axes[2]
    current_probs_row = mean_probs[current_regime]
    yerr = np.vstack([
        current_probs_row - lower_probs_current,
        upper_probs_current - current_probs_row
    ])
    bars = ax3.bar(range(actual_num_regimes), current_probs_row, yerr=yerr, 
                  color=colors, alpha=0.7, capsize=5)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{current_probs_row[i]:.3f}',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    ax3.set_xlabel('To Regime', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    current_regime_desc = regime_descriptions_short.get(current_regime, f"Reg {current_regime}")
    ax3.set_title(f'Transition Probabilities from Current Regime {current_regime} ({current_regime_desc})', fontsize=14)
    ax3.set_xticks(range(actual_num_regimes))
    ax3.set_xticklabels([regime_descriptions_short.get(i, f"Reg {i}") for i in range(actual_num_regimes)])
    ax3.grid(True, alpha=0.3)
    
    stay_prob = current_probs_row[current_regime]
    text_color = 'white' if stay_prob > 0.3 else 'black'
    ax3.text(current_regime, stay_prob/2, f'{stay_prob:.3f}', 
             ha='center', va='center', color=text_color, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'{symbol}_gmm_regimes_{timestamp}.png'), dpi=300)
    plt.show()

def create_transition_network(symbol, transition_probs, regime_info, current_regime, save_dir=None):
    """Create a network diagram visualizing the transition probabilities between regimes."""
    try:
        import networkx as nx
        
        G = nx.DiGraph()
        actual_num_regimes = transition_probs.shape[0]
        
        for i in range(actual_num_regimes):
            G.add_node(i, volatility=regime_info['means'][i] if i < len(regime_info['means']) else np.nan)
        
        for i in range(actual_num_regimes):
            for j in range(actual_num_regimes):
                if transition_probs[i, j] > 0.001: # Threshold to avoid clutter
                    G.add_edge(i, j, weight=transition_probs[i, j])
        
        if not G.nodes:  # No nodes, cannot draw
            print(f"No nodes for transition network for {symbol}. Skipping.")
            return

        plt.figure(figsize=(10, 8))
        pos = nx.circular_layout(G)
        
        node_sizes = [2000 * (regime_info['weights'][i] if i < len(regime_info['weights']) and not pd.isna(regime_info['weights'][i]) else 0.1) for i in G.nodes()]
        
        max_mean_vol = max(m for m in regime_info['means'] if not pd.isna(m)) if any(not pd.isna(m) for m in regime_info['means']) else 1.0
        node_colors_values = [(regime_info['means'][i] if i < len(regime_info['means']) and not pd.isna(regime_info['means'][i]) else 0) / max_mean_vol for i in G.nodes()]
        node_colors = plt.cm.viridis(node_colors_values)
        
        node_border = ['black'] * actual_num_regimes
        if 0 <= current_regime < actual_num_regimes:
            node_border[current_regime] = 'red'
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                             edgecolors=[node_border[n] for n in G.nodes()], linewidths=2) # Ensure border aligns with G.nodes()
        
        edge_weights_viz = [5 * G[u][v]['weight'] for u, v in G.edges()]
        edge_colors_viz = [plt.cm.plasma(G[u][v]['weight']) for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, width=edge_weights_viz, edge_color=edge_colors_viz, 
                             connectionstyle='arc3,rad=0.1', arrowsize=20, alpha=0.7)
        
        regime_descriptions_short = {
            0: "Low Vol", 1: "Med Vol" if actual_num_regimes > 2 else "High Vol",
            2: "High Vol" if actual_num_regimes > 2 else None
        }
        labels = {i: f"R{i}\n{regime_descriptions_short.get(i, '')}\nVol: {regime_info['means'][i]:.2f}" 
                  if i < len(regime_info['means']) and not pd.isna(regime_info['means'][i]) else f"R{i}" 
                  for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
        
        edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)
        
        plt.title(f"{symbol}: Volatility Regime Transition Network (Current: R{current_regime})", fontsize=14)
        plt.axis('off')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(save_dir, f'{symbol}_transition_network_{timestamp}.png'), dpi=300)
        plt.show()
        
    except ImportError:
        print("NetworkX not installed. Skipping transition network. Install with: pip install networkx")
    except Exception as e:
        print(f"Error creating transition network for {symbol}: {e}")

def map_realized_vol_to_implied_vol(realized_vol, regime, days_to_expiration):
    # Historical relationship between realized and implied vol by regime
    regime_vol_premium = {
        'Low': 1.15,     # Implied vol typically 15% higher than realized in low regimes
        'Medium': 1.08,  # 8% premium in medium regimes
        'High': 0.95     # Implied vol often below realized in high regimes
    }
    
    # Term structure adjustment based on regime
    term_adjustment = {
        'Low': 0.03 * math.log(days_to_expiration),
        'Medium': 0.01 * math.log(days_to_expiration),
        'High': -0.02 * math.log(days_to_expiration)
    }
    
    expected_implied_vol = realized_vol * regime_vol_premium[regime] + term_adjustment[regime]
    return expected_implied_vol
def run_gmm_analysis(folder_path=DATA_PATH, window=20, num_regimes=3, prior_strength=1.0, 
                   save_dir="volatility_results_gmm", start_date='2021-01-01', end_date='2025-05-14'):
    print(f"Loading stock data from {folder_path}...")
    all_data = load_stock_data(folder_path)
    
    if not all_data:
        print("No valid stock data found in the specified folder.")
        return {}, []
    
    print(f"Found {len(all_data)} valid stock datasets.")
    print(f"Analysis period: {start_date} to {end_date}")
    
    results_dir = save_dir
    os.makedirs(results_dir, exist_ok=True)
    summary_data = []
    all_results = {}
    
    for symbol, data_df in all_data.items(): # Renamed data to data_df to avoid conflict
        results = analyze_stock_with_gmm(symbol, data_df, window, num_regimes, prior_strength, 
                                      start_date, end_date)
        
        if results:
            all_results[symbol] = results
            # Pass start_date and end_date (strings) to visualization
            visualize_gmm_results(results, results_dir, start_date_str=start_date, end_date_str=end_date)
            create_transition_network(symbol, results['transition_probabilities'], 
                                     results['regime_info'], results['current_regime'], 
                                     save_dir=results_dir)
            
            current_regime = results['current_regime']
            transition_probs = results['transition_probabilities'][current_regime]
            regime_info = results['regime_info']
            actual_num_regimes_found_summary = results['transition_probabilities'].shape[0]


            summary_entry = {
                'Symbol': symbol,
                'Current_Regime': current_regime,
                'Current_Volatility': results['volatility'].dropna().iloc[-1] if not results['volatility'].dropna().empty else None,
                'Regime_Mean_Volatility': regime_info['means'][current_regime] if current_regime < len(regime_info['means']) else np.nan,
                'Prob_Stay_Same': transition_probs[current_regime] if current_regime < len(transition_probs) else np.nan,
            }
            for i in range(actual_num_regimes_found_summary):
                summary_entry[f'Prob_To_Regime_{i}'] = transition_probs[i] if i < len(transition_probs) else np.nan
            for i in range(len(regime_info['means'])): # Use length of means from GMM
                 summary_entry[f'Regime_{i}_Mean'] = regime_info['means'][i]
                 summary_entry[f'Regime_{i}_StdDev'] = np.sqrt(regime_info['variances'][i]) if not pd.isna(regime_info['variances'][i]) else np.nan
                 summary_entry[f'Regime_{i}_Weight'] = regime_info['weights'][i]
            summary_data.append(summary_entry)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.sort_values('Symbol', inplace=True)
        summary_path = os.path.join(results_dir, 'gmm_volatility_regime_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print("\nAnalysis Summary (selected columns):")
        print(summary_df[['Symbol', 'Current_Regime', 'Current_Volatility', 'Regime_Mean_Volatility', 'Prob_Stay_Same']])
    
    return all_results, summary_data

if __name__ == "__main__":
    print("Bayesian Volatility Regime Analysis with GMM Clustering\n")
    
    window_size = 20
    num_volatility_regimes = 3
    prior_strength = 1.0
    start_date = '2021-01-01'
    end_date = '2025-05-14'
    
    print(f"Analysis Parameters:")
    print(f"- Volatility window: {window_size} days")
    print(f"- Number of regimes: {num_volatility_regimes}")
    print(f"- Prior strength: {prior_strength}")
    print(f"- Analysis period: {start_date} to {end_date}")
    
    results, summary = run_gmm_analysis(
        folder_path=DATA_PATH,
        window=window_size,
        num_regimes=num_volatility_regimes,
        prior_strength=prior_strength,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\nAnalysis complete!")