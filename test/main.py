import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy.stats as stats
from datetime import datetime

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
            
            # Handle different date column formats
            date_cols = [col for col in df.columns if any(date_str in col.lower() 
                                                       for date_str in ['date', 'time'])]
            
            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                except:
                    print(f"Could not parse dates in {symbol}, skipping date conversion")
            
            # Make sure it has the columns we need
            if 'close' in df.columns:
                all_data[symbol] = df
                print(f"Successfully loaded {symbol} with {len(df)} rows")
            else:
                # Try to find the close price column
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
    # Calculate returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate rolling standard deviation (annualized)
    volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

def identify_volatility_regimes(volatility, num_regimes=3):
    """Identify volatility regimes using simple quantile thresholds."""
    # Drop NaN values for quantile calculation
    clean_vol = volatility.dropna()
    
    if clean_vol.empty:
        print("Warning: No valid volatility data for regime identification")
        return pd.Series(index=volatility.index)
    
    if num_regimes == 2:
        # Simple high/low regime
        threshold = np.median(clean_vol)
        regimes = pd.Series(0, index=volatility.index)
        regimes[volatility > threshold] = 1
    else:
        # Multiple regimes based on quantiles
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
        return np.zeros((1, 1))
    
    unique_regimes = sorted(regimes_clean.unique().astype(int))
    num_regimes = len(unique_regimes)
    
    # Initialize transition count matrix
    transition_counts = np.zeros((num_regimes, num_regimes))
    
    # Count transitions
    prev_regime = None
    for regime in regimes_clean:
        regime = int(regime)
        if prev_regime is not None:
            transition_counts[prev_regime, regime] += 1
        prev_regime = regime
        
    return transition_counts

def bayesian_transition_analysis(transition_counts, prior_strength=1.0):
    """
    Perform Bayesian analysis of regime transitions using Dirichlet-Multinomial model.
    """
    num_regimes = transition_counts.shape[0]
    if num_regimes == 0:
        print("Warning: No regimes identified for Bayesian analysis")
        return np.array([]), {}, {}
    
    # Initialize storage for results
    posterior_samples = {}
    mean_probs = np.zeros_like(transition_counts, dtype=float)
    
    # For each starting regime, calculate posterior distribution
    for from_regime in range(num_regimes):
        # Prior: Uniform Dirichlet (all alphas = prior_strength)
        prior_alphas = np.ones(num_regimes) * prior_strength
        
        # Add counts to create posterior
        posterior_alphas = prior_alphas + transition_counts[from_regime]
        
        # Generate samples from posterior
        samples = np.random.default_rng().dirichlet(posterior_alphas, size=10000)
        posterior_samples[from_regime] = samples
        
        # Mean probability (expected value of Dirichlet)
        mean_probs[from_regime] = posterior_alphas / posterior_alphas.sum()
    
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

def analyze_stock(symbol, data, window=20, num_regimes=3, prior_strength=1.0):
    """Analyze volatility regimes for a single stock."""
    print(f"\nAnalyzing {symbol}...")
    
    # Calculate volatility
    volatility = calculate_volatility(data, window)
    
    # Check if we have enough data
    if volatility.dropna().empty:
        print(f"Not enough data for {symbol} to calculate volatility")
        return None
        
    # Identify regimes
    regimes = identify_volatility_regimes(volatility, num_regimes)
    
    # Count transitions
    transition_counts = count_transitions(regimes)
    
    # Bayesian analysis
    mean_probs, posterior_samples = bayesian_transition_analysis(
        transition_counts, prior_strength)
    
    if len(posterior_samples) == 0:
        print(f"Not enough regime transitions for {symbol}")
        return None
    
    # Calculate credible intervals
    lower_probs, upper_probs = calculate_probability_intervals(posterior_samples)
    
    # Find current regime (most recent non-NaN)
    current_regime = None
    if not regimes.dropna().empty:
        current_regime = int(regimes.dropna().iloc[-1])
    else:
        print(f"No valid current regime for {symbol}")
        return None
    
    # Prepare results
    results = {
        'symbol': symbol,
        'volatility': volatility,
        'regimes': regimes,
        'transition_counts': transition_counts,
        'transition_probabilities': mean_probs,
        'posterior_samples': posterior_samples,
        'lower_probabilities': lower_probs,
        'upper_probabilities': upper_probs,
        'current_regime': current_regime
    }
    
    # Print key results
    print(f"Current volatility regime: {current_regime}")
    print("\nTransition probability matrix:")
    df_probs = pd.DataFrame(
        mean_probs, 
        index=[f'From Regime {i}' for i in range(num_regimes)],
        columns=[f'To Regime {i}' for i in range(num_regimes)]
    )
    print(df_probs)
    
    print(f"\nProbabilities of transitioning from current regime {current_regime}:")
    current_probs = mean_probs[current_regime]
    current_lower = lower_probs[current_regime]
    current_upper = upper_probs[current_regime]
    
    for to_regime in range(num_regimes):
        print(f"  To Regime {to_regime}: {current_probs[to_regime]:.3f} " +
              f"(95% CI: {current_lower[to_regime]:.3f} - {current_upper[to_regime]:.3f})")
    
    return results

def visualize_results(results, save_dir=None):
    """Create visualization of the volatility regimes and transition probabilities."""
    symbol = results['symbol']
    volatility = results['volatility']
    regimes = results['regimes']
    mean_probs = results['transition_probabilities']
    current_regime = results['current_regime']
    lower_probs = results['lower_probabilities'][current_regime]
    upper_probs = results['upper_probabilities'][current_regime]
    
    num_regimes = mean_probs.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 2, 1.5]})
    
    # Plot 1: Volatility with regimes
    ax1 = axes[0]
    volatility.plot(ax=ax1, color='black', alpha=0.5, label='Volatility')
    
    # Color each regime differently
    colors = plt.cm.tab10(np.linspace(0, 1, num_regimes))
    for i in range(num_regimes):
        mask = regimes == i
        ax1.scatter(volatility.index[mask], volatility[mask], 
                    color=colors[i], label=f'Regime {i}', alpha=0.7)
    
    ax1.set_title(f'{symbol} - Volatility Regimes', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime over time
    ax2 = axes[1]
    # Plot regimes as a step function
    regimes.plot(ax=ax2, drawstyle='steps-post', linewidth=2)
    
    # Color the background based on regime
    for i in range(num_regimes):
        mask = regimes == i
        if mask.any():
            # Find continuous periods
            mask = mask.astype(int)
            changes = mask.diff().fillna(0).astype(bool)
            start_idx = mask.index[changes & (mask == 1)]
            end_idx = mask.index[changes.shift(-1, fill_value=False) & (mask == 1)]
            
            # If the last period extends to the end
            if len(start_idx) > len(end_idx):
                end_idx = pd.Index([mask.index[-1]]).append(end_idx)
                
            # Fill each period
            for start, end in zip(start_idx, end_idx):
                ax2.axvspan(start, end, alpha=0.2, color=colors[i])
    
    ax2.set_yticks(range(num_regimes))
    ax2.set_title(f'{symbol} - Volatility Regime Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Transition probabilities from current regime
    ax3 = axes[2]
    current_probs = mean_probs[current_regime]
    
    # Calculate error bars
    yerr = np.vstack([
        current_probs - lower_probs,
        upper_probs - current_probs
    ])
    
    bars = ax3.bar(range(num_regimes), current_probs, yerr=yerr, 
                  color=colors, alpha=0.7, capsize=5)
    
    # Add probability labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{current_probs[i]:.3f}',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    ax3.set_xlabel('To Regime', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title(f'Transition Probabilities from Current Regime {current_regime}', fontsize=14)
    ax3.set_xticks(range(num_regimes))
    ax3.set_xticklabels([f'Regime {i}' for i in range(num_regimes)])
    ax3.grid(True, alpha=0.3)
    
    # Stay probability annotation
    stay_prob = current_probs[current_regime]
    text_color = 'white' if stay_prob > 0.3 else 'black'
    ax3.text(current_regime, stay_prob/2, f'{stay_prob:.3f}', 
             ha='center', va='center', color=text_color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure if a directory is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'{symbol}_volatility_regimes_{timestamp}.png'), dpi=300)
    
    plt.show()

def run_analysis(folder_path=DATA_PATH, window=20, num_regimes=3, prior_strength=1.0, 
                save_dir="volatility_results"):
    """Run analysis on all stock data in folder."""
    print(f"Loading stock data from {folder_path}...")
    all_data = load_stock_data(folder_path)
    
    if not all_data:
        print("No valid stock data found in the specified folder.")
        return
    
    print(f"Found {len(all_data)} valid stock datasets.")
    
    # Create results directory
    results_dir = save_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a summary dataframe
    summary_data = []
    
    all_results = {}
    
    for symbol, data in all_data.items():
        results = analyze_stock(symbol, data, window, num_regimes, prior_strength)
        
        if results:
            all_results[symbol] = results
            visualize_results(results, results_dir)
            
            # Add to summary
            current_regime = results['current_regime']
            transition_probs = results['transition_probabilities'][current_regime]
            
            summary_data.append({
                'Symbol': symbol,
                'Current_Regime': current_regime,
                'Prob_Stay_Same': transition_probs[current_regime],
                **{f'Prob_To_Regime_{i}': prob for i, prob in enumerate(transition_probs)}
            })
    
    # Create and save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.sort_values('Symbol', inplace=True)
        
        # Save summary to CSV
        summary_path = os.path.join(results_dir, 'volatility_regime_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        
        # Print summary
        print("\nAnalysis Summary:")
        print(summary_df)
    
    return all_results, summary_data

if __name__ == "__main__":
    print("Bayesian Volatility Regime Analysis\n")
    
    # Parameters
    window_size = 20          # Window for volatility calculation (trading days)
    num_volatility_regimes = 3  # Number of regimes to identify
    prior_strength = 1.0      # Prior strength (higher = more conservative)
    
    print(f"Analysis Parameters:")
    print(f"- Volatility window: {window_size} days")
    print(f"- Number of regimes: {num_volatility_regimes}")
    print(f"- Prior strength: {prior_strength}")
    
    # Run analysis
    results, summary = run_analysis(
        folder_path=DATA_PATH,
        window=window_size,
        num_regimes=num_volatility_regimes,
        prior_strength=prior_strength
    )
    
    print("\nAnalysis complete!")