"""
Visualization functions for market regime analysis.

This module provides a set of plotting functions to visualize various
aspects of market regimes identified by Bayesian HMM models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
# Removed: import pandas as pd (no longer used)
# PercentFormatter is not used, can be removed if not planned for future use
# from matplotlib.ticker import PercentFormatter


def _create_color_palette(n_regimes):
    """Create a consistent color palette for regime visualization"""
    if n_regimes <= 0: # Handle case with no regimes
        return []
    if n_regimes <= 3:
        return ['#2ecc71', '#f1c40f', '#e74c3c'][:n_regimes]
    elif n_regimes <= 5:
        return ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c'][:n_regimes]
    else:
        return sns.color_palette("viridis", n_regimes)


def plot_regime_distribution(regime_stats, figsize=(12, 10), output_path=None):
    """
    Create visualizations showing the distribution of regimes
    
    Parameters:
    -----------
    regime_stats : list
        List of dictionaries containing regime statistics
    figsize : tuple
        Figure size (width, height) in inches
    output_path : str, optional
        If provided, the plot will be saved to this path
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    if not regime_stats:
        print("Warning: regime_stats is empty. Cannot plot regime distribution.")
        axes[0].text(0.5, 0.5, "No data for regime distribution.", ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No data for regime duration.", ha='center', va='center', transform=axes[1].transAxes)
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig

    regimes = [stat['regime'] for stat in regime_stats]
    counts = [stat['count'] for stat in regime_stats]
    percentages = [stat['percentage'] for stat in regime_stats]
    labels = [f"Regime {stat['regime']}: {stat['description']}" for stat in regime_stats]
    
    colors = _create_color_palette(len(regimes))
    if not colors and percentages: # If colors are empty but percentages exist
        colors = sns.color_palette("viridis", len(percentages))


    axes[0].pie(
        percentages, 
        labels=None,
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    axes[0].set_title('Percentage of Time in Each Regime', fontsize=14)
    axes[0].legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)
    
    bar_plot = sns.barplot(x=regimes, y=counts, palette=colors, ax=axes[1])
    
    for i, p in enumerate(bar_plot.patches):
        bar_plot.annotate(
            f'{counts[i]}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10
        )
    
    axes[1].set_xlabel('Regime', fontsize=12)
    axes[1].set_ylabel('Number of Days', fontsize=12)
    axes[1].set_title('Duration of Each Regime', fontsize=14)
    axes[1].set_xticklabels([f'Regime {r}' for r in regimes])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved regime distribution plot to {output_path}")
    
    return fig


def plot_regime_timeline(dates, regimes, regime_stats, ohlcv=None, figsize=(16, 10), output_path=None):
    """
    Plot regime timeline with price overlay
    
    Parameters:
    -----------
    dates : list
        List of dates
    regimes : list
        List of regime assignments
    regime_stats : list
        List of dictionaries containing regime statistics
    ohlcv : dict, optional
        Dictionary with OHLCV data. If provided, price will be overlaid ('close' key expected).
    figsize : tuple
        Figure size (width, height) in inches
    output_path : str, optional
        If provided, the plot will be saved to this path
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Print debug info for original lengths
    print(f"Debug - Plot Timeline: initial dates={len(dates)}, initial regimes={len(regimes)}")
    
    # Make sure lengths match by finding the minimum length
    min_length = min(len(dates), len(regimes))
    dates = dates[:min_length]
    regimes = regimes[:min_length]
    
    fig, ax1 = plt.subplots(figsize=figsize) # Create figure early for potential empty return

    if not dates: # After truncation, dates could be empty
        print("Warning: Dates array is empty after truncation. Cannot plot timeline.")
        ax1.text(0.5, 0.5, "No data to display for timeline.", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Market Regimes Over Time (No Data)', fontsize=16)
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved empty regime timeline plot to {output_path}")
        return fig

    # Convert dates to datetime if they're strings
    if dates and isinstance(dates[0], str): # check if dates is not empty before accessing dates[0]
        try:
            date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        except ValueError as e:
            print(f"Error converting dates in plot_regime_timeline: {e}. Aborting plot.")
            ax1.text(0.5, 0.5, "Error in date format.", ha='center', va='center', transform=ax1.transAxes)
            if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig
    else:
        date_objects = dates # Assume they are already datetime objects or list is empty
    
    sns.set_style("white")
    
    if not regimes:
        n_regimes_data = 0
    else:
        unique_regimes_in_data = sorted(list(set(r for r in regimes if r is not None))) # Filter out Nones
        if not unique_regimes_in_data:
             n_regimes_data = 0
        else:
             n_regimes_data = max(unique_regimes_in_data) + 1
             
    colors = _create_color_palette(n_regimes_data)
    
    ax2 = None # Initialize ax2
    if ohlcv is not None and 'close' in ohlcv:
        ax2 = ax1.twinx()
    
    changes = []
    if regimes: 
        current_regime = regimes[0]
        current_start_idx = 0
        for i, r_val in enumerate(regimes):
            if r_val != current_regime:
                changes.append((current_start_idx, i - 1, current_regime))
                current_regime = r_val
                current_start_idx = i
        changes.append((current_start_idx, len(regimes) - 1, current_regime))

    plotted_labels = set() 
    for start_idx, end_idx, r_val in changes:
        if r_val is None: continue # Skip if regime is None
        label = f'Regime {r_val}'
        color_idx = r_val % len(colors) if colors else 0
        current_color = colors[color_idx] if colors else 'gray'

        if label not in plotted_labels:
            ax1.axvspan(
                date_objects[start_idx], 
                date_objects[end_idx], 
                alpha=0.3, 
                color=current_color,
                label=label
            )
            plotted_labels.add(label)
        else:
            ax1.axvspan(
                date_objects[start_idx], 
                date_objects[end_idx], 
                alpha=0.3, 
                color=current_color
            )
            
    handles, labels_from_plot = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels_from_plot, handles))
    
    if ax2 and ohlcv is not None and 'close' in ohlcv: # Check ax2 existence
        close_prices_orig = ohlcv['close']
        close_prices_final = []

        if len(close_prices_orig) == min_length:
            close_prices_final = close_prices_orig
        elif len(close_prices_orig) > min_length:
            close_prices_final = close_prices_orig[:min_length]
        else: 
            _cp_list = list(close_prices_orig)
            if not _cp_list and min_length > 0:
                close_prices_final = [np.nan] * min_length
            elif _cp_list: 
                close_prices_final = _cp_list + [_cp_list[-1]] * (min_length - len(_cp_list))
            
        if close_prices_final: 
            ax2.plot(date_objects, close_prices_final, color='black', linewidth=1.5)
            ax2.set_ylabel('Price', fontsize=12)
        elif ax2: # if ax2 was created
             ax2.set_ylabel('Price (No data)', fontsize=12)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Regime', fontsize=12)
    ax1.set_title('Market Regimes Over Time', fontsize=16)
    
    legend_handles_final = []
    legend_labels_final = []
    
    # Get actually plotted regime numbers to build legend correctly
    actual_plotted_regime_nums = sorted([int(lbl.split()[-1]) for lbl in by_label.keys()])

    for regime_num in actual_plotted_regime_nums:
        regime_plot_label = f"Regime {regime_num}"
        if regime_plot_label not in by_label: continue # Should not happen if logic is correct

        handle = by_label[regime_plot_label]
        description = f"Regime {regime_num}" 
        for stat in regime_stats:
            if stat['regime'] == regime_num:
                description = f"Regime {stat['regime']}: {stat['description']}"
                break
        
        legend_handles_final.append(handle)
        legend_labels_final.append(description)
        
    if legend_handles_final:
        ax1.legend(legend_handles_final, legend_labels_final, loc='upper left', frameon=True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved regime timeline plot to {output_path}")
    
    return fig


def plot_regime_characteristics(regime_stats, figsize=(14, 12), output_path=None):
    """
    Plot radar charts showing characteristics of each regime
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    if not regime_stats:
        print("Warning: regime_stats is empty. Cannot plot regime characteristics.")
        ax.text(0, 0, "No data for regime characteristics.", ha='center', va='center') 
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig

    n_regimes = len(regime_stats)
    colors = _create_color_palette(n_regimes)
    if not colors and n_regimes > 0:
        colors = sns.color_palette("viridis", n_regimes)

    characteristics_keys = ['mean_return', 'volatility', 'sharpe', 'mean_atr']
    char_labels_map = {'mean_return': 'Return', 'volatility': 'Volatility', 'sharpe': 'Sharpe Ratio', 'mean_atr': 'ATR'}
    
    valid_characteristics = []
    for char_key in characteristics_keys:
        if all(char_key in stat and stat[char_key] is not None for stat in regime_stats): # Check for None as well
            valid_characteristics.append(char_key)
        else:
            print(f"Warning: Characteristic '{char_key}' missing or None in some regime_stats. Skipping it for radar plot.")
    
    if not valid_characteristics:
        print("Warning: No valid characteristics with non-None values to plot for radar chart.")
        ax.text(0, 0, "No valid characteristics to plot.", ha='center', va='center')
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig

    N = len(valid_characteristics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    max_values = {}
    min_values = {}
    for char in valid_characteristics:
        values = [stat[char] for stat in regime_stats] 
        max_val, min_val = max(values), min(values)
        if max_val == min_val:
            min_val_eff = max_val * 0.9 if max_val != 0 else -0.1
            max_val_eff = max_val * 1.1 if max_val != 0 else 0.1
            if max_val_eff == min_val_eff : 
                max_val_eff = min_val_eff + 1 
            max_values[char], min_values[char] = max_val_eff, min_val_eff
        else:
            max_values[char], min_values[char] = max_val, min_val
    
    for i, stat in enumerate(regime_stats):
        scaled_values = []
        for char in valid_characteristics:
            val = stat[char]
            if max_values[char] == min_values[char]: 
                scaled_values.append(0.5) 
            else:
                scaled_values.append(np.clip((val - min_values[char]) / (max_values[char] - min_values[char]), 0, 1)) # Clip to [0,1]
        
        scaled_values += scaled_values[:1]
        ax.plot(angles, scaled_values, color=colors[i % len(colors)], linewidth=2, label=f"Regime {stat.get('regime','N/A')}: {stat.get('description','No desc.')}")
        ax.fill(angles, scaled_values, color=colors[i % len(colors)], alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([char_labels_map.get(char, char) for char in valid_characteristics])
    ax.set_rlabel_position(0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['Min', '25%', '50%', '75%', 'Max']) 
    ax.set_ylim(0, 1)
    
    plt.legend(loc='best', bbox_to_anchor=(1.1, 1.0)) # Adjust legend position
    plt.title('Regime Characteristics Comparison', fontsize=16, y=1.08)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved regime characteristics plot to {output_path}")
    return fig


def plot_transition_heatmap(hmm, figsize=(12, 10), output_path=None):
    """
    Plot a heatmap of transition probabilities between regimes
    
    Parameters:
    -----------
    hmm : BayesianHMM
        Fitted Bayesian HMM model. Expects hmm.transmat_ to exist.
    figsize : tuple
        Figure size (width, height) in inches
    output_path : str, optional
        If provided, the plot will be saved to this path
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)

    if not hasattr(hmm, 'transmat_') or hmm.transmat_ is None:
        msg = "HMM model does not have 'transmat_' attribute or it's None."
        print(f"Warning: {msg} Cannot plot transition heatmap.")
        ax.text(0.5, 0.5, "Transition matrix unavailable.", ha='center', va='center', transform=ax.transAxes)
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig

    raw_transmat = hmm.transmat_
    transition_matrix = None

    if isinstance(raw_transmat, list):
        try:
            transition_matrix = np.array(raw_transmat)
        except Exception as e:
            msg = f"Error converting transition matrix from list to numpy array: {e}"
            print(f"Warning: {msg}")
            ax.text(0.5, 0.5, "Error converting transition matrix.", ha='center', va='center', transform=ax.transAxes)
            if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig
    elif isinstance(raw_transmat, np.ndarray):
        transition_matrix = raw_transmat
    else:
        msg = f"hmm.transmat_ is of unexpected type: {type(raw_transmat)}."
        print(f"Warning: {msg} Cannot plot transition heatmap.")
        ax.text(0.5, 0.5, "Transition matrix has unexpected type.", ha='center', va='center', transform=ax.transAxes)
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Validate the resulting transition_matrix
    if transition_matrix is None or not isinstance(transition_matrix, np.ndarray) or \
       transition_matrix.ndim != 2 or transition_matrix.shape[0] == 0 or \
       transition_matrix.shape[0] != transition_matrix.shape[1]:
        shape_info = getattr(transition_matrix, 'shape', 'N/A (None or not an array)')
        msg = f"Transition matrix is not a valid square 2D numpy array. Shape: {shape_info}."
        print(f"Warning: {msg} Cannot plot transition heatmap.")
        ax.text(0.5, 0.5, "Invalid transition matrix format.", ha='center', va='center', transform=ax.transAxes)
        if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig

    cmap = LinearSegmentedColormap.from_list('blue_white', ['#ffffff', '#3498db'])
    
    sns.heatmap(
        transition_matrix,
        annot=True,
        cmap=cmap,
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Transition Probability'}
    )
    
    ax.set_xlabel('To Regime', fontsize=12)
    ax.set_ylabel('From Regime', fontsize=12)
    ax.set_title('Regime Transition Probabilities', fontsize=16)
    
    n_regimes = transition_matrix.shape[0] # This should now work
    ax.set_xticklabels([f'Regime {i}' for i in range(n_regimes)])
    ax.set_yticklabels([f'Regime {i}' for i in range(n_regimes)])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition heatmap to {output_path}")
    
    return fig


def plot_performance_metrics(regime_stats, features, regimes, dates, figsize=(14, 10), output_path=None):
    """
    Plot performance metrics for each regime
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    if features is None or 'log_returns' not in features or not features['log_returns']:
        msg = "Features data is missing, 'log_returns' not in features, or log_returns is empty."
        print(f"Warning: {msg} Cannot plot performance metrics.")
        for i, ax_single in enumerate(axes):
            ax_single.text(0.5, 0.5, f"Plot {i+1}: Data unavailable.", ha='center', va='center', transform=ax_single.transAxes)
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
        
    returns_orig = features['log_returns']
    
    print(f"Debug - Plot Performance: initial returns={len(returns_orig)}, initial regimes={len(regimes)}, initial dates={len(dates)}")

    min_length = min(len(returns_orig), len(regimes), len(dates))

    if min_length == 0:
        msg = "One or more input arrays (returns, regimes, dates) are effectively empty after considering common length."
        print(f"Warning: {msg} Cannot plot performance metrics.")
        for i, ax_single in enumerate(axes):
            ax_single.text(0.5, 0.5, f"Plot {i+1}: Insufficient data.", ha='center', va='center', transform=ax_single.transAxes)
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig

    returns = np.array(returns_orig[:min_length]) # Ensure returns is a numpy array for easier ops
    regimes_truncated = regimes[:min_length]
    dates_truncated = dates[:min_length]

    date_objects = dates_truncated 
    if dates_truncated and isinstance(dates_truncated[0], str):
        try:
            date_objects = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in dates_truncated]
        except ValueError as e:
            print(f"Error converting dates in plot_performance_metrics: {e}. Date-axis plots may fail.")
            # Keep date_objects as original strings, plotting might fail or look odd.
    
    n_regimes_from_stats = len(regime_stats)
    colors = _create_color_palette(n_regimes_from_stats)
    if not colors and n_regimes_from_stats > 0:
         colors = sns.color_palette("viridis", n_regimes_from_stats)


    # 1. Return distribution histograms
    for i, stat in enumerate(regime_stats):
        regime_val = stat['regime']
        # Use boolean indexing with numpy array for returns
        regime_mask = np.array([r == regime_val for r in regimes_truncated])
        regime_returns_for_hist = returns[regime_mask]

        if not regime_returns_for_hist.any(): continue
        sns.histplot(
            regime_returns_for_hist, color=colors[i % len(colors)], kde=True, ax=axes[0],
            alpha=0.3, stat="density", label=f"Regime {regime_val}"
        )
    axes[0].set_xlabel('Log Returns', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Return Distributions by Regime', fontsize=14)
    if any(col.get_visible() for col in axes[0].collections) or any(line.get_visible() for line in axes[0].lines):
        axes[0].legend()
    
    # 2. Box plots of returns
    regime_returns_data_for_boxplot = []
    regime_labels_for_boxplot = []
    
    unique_regimes_in_data = sorted(list(set(r for r in regimes_truncated if r is not None)))

    for r_idx, regime_val in enumerate(unique_regimes_in_data):
        regime_mask = np.array([r == regime_val for r in regimes_truncated])
        current_regime_returns = returns[regime_mask]
        if current_regime_returns.any(): 
            regime_returns_data_for_boxplot.append(current_regime_returns)
            regime_labels_for_boxplot.append(f'Regime {regime_val}')

    if regime_returns_data_for_boxplot:
        bp = axes[1].boxplot(
            regime_returns_data_for_boxplot, labels=regime_labels_for_boxplot, patch_artist=True,
            boxprops=dict(facecolor='lightgrey'), flierprops=dict(markerfacecolor='grey', marker='o', markersize=4)
        )
        if colors and len(colors) >= len(unique_regimes_in_data):
            stat_regime_to_idx_map = {stat['regime']: idx for idx, stat in enumerate(regime_stats)}
            for patch_idx, regime_label_str in enumerate(regime_labels_for_boxplot):
                regime_num = int(regime_label_str.split()[-1])
                color_stat_idx = stat_regime_to_idx_map.get(regime_num, regime_num) # Fallback to regime_num if not in stats
                bp['boxes'][patch_idx].set_facecolor(colors[color_stat_idx % len(colors)])

    axes[1].set_xlabel('Regime', fontsize=12)
    axes[1].set_ylabel('Log Returns', fontsize=12)
    axes[1].set_title('Return Volatility by Regime', fontsize=14)

    # 3. Sharpe ratio and other metrics bar chart
    metrics_keys = ['mean_return', 'sharpe'] 
    metric_names_display = ['Annualized Return', 'Sharpe Ratio']
    x_bar = np.arange(len(metric_names_display))
    num_bars_total = n_regimes_from_stats
    bar_width = 0.8 / num_bars_total if num_bars_total > 0 else 0.8
    
    for i, stat in enumerate(regime_stats):
        regime_val = stat['regime']
        mean_ret_annualized = stat.get('mean_return', 0) * 252
        sharpe_val = stat.get('sharpe', 0)
        values_for_bar = [mean_ret_annualized, sharpe_val]
        
        offset = bar_width * i - (bar_width * (num_bars_total - 1)) / 2 if num_bars_total > 0 else 0
        axes[2].bar(x_bar + offset, values_for_bar, bar_width, color=colors[i % len(colors)], label=f'Regime {regime_val}')
    
    axes[2].set_xlabel('Metric', fontsize=12)
    axes[2].set_ylabel('Value', fontsize=12)
    axes[2].set_title('Risk-Adjusted Performance by Regime', fontsize=14)
    axes[2].set_xticks(x_bar)
    axes[2].set_xticklabels(metric_names_display)
    if any(patch.get_visible() for patch in axes[2].patches): axes[2].legend()

    # 4. Cumulative returns by regime
    if date_objects and len(date_objects) == min_length: # Ensure date_objects are valid and match length
        for i, stat_info in enumerate(regime_stats):
            regime_val = stat_info['regime']
            
            masked_regime_returns = np.where(np.array(regimes_truncated) == regime_val, returns, 0)
            cumulative = np.cumsum(masked_regime_returns)
            
            axes[3].plot(date_objects, cumulative, color=colors[i % len(colors)], label=f'Regime {regime_val}')
        
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Adjust interval based on data length for better readability
        num_months = (date_objects[-1] - date_objects[0]).days / 30.44 if len(date_objects) > 1 else 1
        locator_interval = max(1, int(num_months / 4)) # Aim for approx 4-5 ticks
        axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=locator_interval)) 
        plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha="right")
        if any(line.get_visible() for line in axes[3].lines): axes[3].legend()

    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].set_ylabel('Cumulative Log Return', fontsize=12)
    axes[3].set_title('Cumulative Returns by Regime', fontsize=14)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance metrics plot to {output_path}")
    
    return fig