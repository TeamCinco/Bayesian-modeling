"""
Interactive dashboard generator for market regime analysis.

This module provides functionality to create interactive dashboards
for visualizing market regimes and their characteristics.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


def _create_color_palette(n_regimes):
    """Create a consistent color palette for regime visualization"""
    if n_regimes <= 0: return [] # Handle case with no regimes
    # Define a consistent color scheme for regimes
    if n_regimes <= 3:
        return ['#2ecc71', '#f1c40f', '#e74c3c'][:n_regimes]
    elif n_regimes <= 5:
        return ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c'][:n_regimes]
    else:
        # For more regimes: Use colorscale
        colorscale = px.colors.sequential.Viridis
        # Ensure we don't go out of bounds for colorscale if n_regimes is large
        return [colorscale[int(i * (len(colorscale)-1) / (n_regimes-1))] if n_regimes > 1 else colorscale[0] for i in range(n_regimes)]


def create_dashboard(results, output_dir, filename='regime_dashboard.html'):
    """
    Create an interactive dashboard for market regime analysis
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary containing regimes, statistics, etc.
    output_dir : str
        Directory to save the dashboard HTML file
    filename : str, optional
        Filename for the dashboard HTML file
        
    Returns:
    --------
    dashboard_path : str
        Path to the created dashboard HTML file
    """
    # Extract data from results
    raw_dates = results['dates']
    raw_regimes = results['regimes']
    regime_stats = results['regime_stats']
    raw_regime_probs = results.get('regime_probs', None)
    raw_ohlcv = results['ohlcv']
    raw_features = results['features']

    # --- Data Alignment ---
    # Primary drivers for length are often regimes, features, and regime_probs
    # These are typically one shorter than raw dates/ohlcv if features involve differences.

    # Length of regimes is a good candidate for the target length.
    # If regimes itself is shorter than dates, it implies it's already processed.
    len_regimes = len(raw_regimes)
    
    # Ensure features and regime_probs match this length, or find a new minimum.
    # It's assumed features['log_returns'] and regime_probs (if present)
    # are already aligned with `raw_regimes` from the HMM processing.
    
    # The number of data points for plotting will be len_regimes.
    # We need to slice raw_dates and raw_ohlcv.
    # Common practice: if returns/regimes start from the 2nd data point,
    # drop the first date and OHLCV point.
    
    if len(raw_dates) > len_regimes:
        # Assuming regimes/features correspond to dates from the second point onwards
        print(f"Debug Dashboard: Aligning dates/OHLCV. Original dates: {len(raw_dates)}, regimes: {len_regimes}. Slicing first {len(raw_dates) - len_regimes} elements from dates/OHLCV.")
        slice_start_index = len(raw_dates) - len_regimes
        dates_aligned = raw_dates[slice_start_index:]
        ohlcv_close_aligned = raw_ohlcv['close'][slice_start_index:]
        ohlcv_volume_aligned = raw_ohlcv['volume'][slice_start_index:]
    elif len(raw_dates) < len_regimes:
        # This case is unusual, implies regimes has more data than dates. Log and adjust.
        print(f"Warning Dashboard: Regimes array ({len_regimes}) is longer than dates array ({len(raw_dates)}). Truncating regimes.")
        raw_regimes = raw_regimes[:len(raw_dates)]
        len_regimes = len(raw_regimes) # update len_regimes
        dates_aligned = raw_dates
        ohlcv_close_aligned = raw_ohlcv['close'] # Assuming ohlcv matches raw_dates
        ohlcv_volume_aligned = raw_ohlcv['volume']
    else: # Lengths match
        dates_aligned = raw_dates
        ohlcv_close_aligned = raw_ohlcv['close']
        ohlcv_volume_aligned = raw_ohlcv['volume']

    # Now, ensure features and regime_probs also align with this new 'len_regimes'
    # `features['log_returns']` should ideally already match `len_regimes`
    # `raw_regime_probs` should also ideally already match `len_regimes`

    log_returns_aligned = raw_features['log_returns']
    if len(log_returns_aligned) != len_regimes:
        print(f"Warning Dashboard: Length of log_returns ({len(log_returns_aligned)}) does not match length of regimes ({len_regimes}). Attempting to align.")
        # This might indicate an issue upstream. For now, truncate/pad cautiously or raise error.
        # Simplest fix: truncate longer array. More robust: re-evaluate upstream logic.
        min_len_for_df = min(len_regimes, len(log_returns_aligned))
        
        dates_aligned = dates_aligned[:min_len_for_df] # Further truncate if needed
        ohlcv_close_aligned = ohlcv_close_aligned[:min_len_for_df]
        ohlcv_volume_aligned = ohlcv_volume_aligned[:min_len_for_df]
        regimes_aligned = raw_regimes[:min_len_for_df]
        log_returns_aligned = log_returns_aligned[:min_len_for_df]
        
        if raw_regime_probs:
            # Check if raw_regime_probs is a list of lists/tuples
            if raw_regime_probs and isinstance(raw_regime_probs[0], (list, tuple, np.ndarray)):
                 raw_regime_probs = [p for p in raw_regime_probs[:min_len_for_df]]
            else: # It might be a dict of lists e.g. {'prob_regime_0': [...]}
                for k in raw_regime_probs.keys():
                    raw_regime_probs[k] = raw_regime_probs[k][:min_len_for_df]
        print(f"Debug Dashboard: Final common length for DataFrame construction: {min_len_for_df}")

    else: # lengths of returns and regimes match
        regimes_aligned = raw_regimes
        min_len_for_df = len_regimes # Use this for regime_probs alignment too
        print(f"Debug Dashboard: Using common length from regimes/returns: {min_len_for_df}")


    # Align regime_probs
    regime_probs_aligned = None
    if raw_regime_probs:
        # Check if raw_regime_probs is a list of lists/tuples (each inner list is probs for a timestep)
        if isinstance(raw_regime_probs, list) and raw_regime_probs and isinstance(raw_regime_probs[0], (list, tuple, np.ndarray)):
            if len(raw_regime_probs) != min_len_for_df:
                print(f"Warning Dashboard: Length of regime_probs list ({len(raw_regime_probs)}) does not match common length ({min_len_for_df}). Truncating.")
                regime_probs_aligned = [p for p in raw_regime_probs[:min_len_for_df]]
            else:
                regime_probs_aligned = raw_regime_probs
        # Check if it's a dict of prob lists (e.g., from a previous version of dashboard)
        elif isinstance(raw_regime_probs, dict):
             # This format is not directly used by the current dashboard.py logic for adding to df
             # The current dashboard.py expects `regime_probs` to be a list of lists/arrays
             # where each inner list contains probabilities for [regime0, regime1, ...] for one time step
             print("Warning Dashboard: regime_probs is a dict, but dashboard expects list of probability arrays. Probabilities might not be shown correctly.")
             # Attempt to convert or handle:
             # If you are sure it's like {'prob_regime_0': [...], 'prob_regime_1': [...]},
             # then this logic needs to be changed when adding to df.
             # For now, we'll assume if it's a dict, it's not the expected format.
             pass # Keep regime_probs_aligned as None, or try to reconstruct below if needed

    # Convert dates to datetime if they're strings
    if dates_aligned and isinstance(dates_aligned[0], str):
        date_objects_aligned = [datetime.strptime(date, '%Y-%m-%d') for date in dates_aligned]
    else:
        date_objects_aligned = dates_aligned # Assume already datetime objects or empty

    # Create a DataFrame for easier plotting
    # All arrays here MUST be of the same length.
    df_dict = {
        'Date': date_objects_aligned,
        'Regime': regimes_aligned,
        'Close': ohlcv_close_aligned,
        'Volume': ohlcv_volume_aligned,
        'Return': log_returns_aligned
    }
    
    # Check lengths before creating DataFrame
    for key, value in df_dict.items():
        print(f"Debug Dashboard - DataFrame input: {key} length = {len(value)}")
        if len(value) != min_len_for_df:
             raise ValueError(f"CRITICAL ERROR in Dashboard: Mismatch for {key}. Expected {min_len_for_df}, got {len(value)}. Upstream data alignment failed.")

    df = pd.DataFrame(df_dict)
    
    # Determine number of regimes based on aligned data for consistency
    # Filter out None before finding max, handle empty or all-None regimes
    valid_regimes_in_df = [r for r in df['Regime'].unique() if r is not None]
    if not valid_regimes_in_df:
        print("Warning Dashboard: No valid regimes found in the aligned data. Some plots may be empty or fail.")
        n_regimes_from_data = 0
    else:
        n_regimes_from_data = max(valid_regimes_in_df) + 1
        if n_regimes_from_data == 0 and regime_stats: # Fallback if regimes are all 0 (e.g. only 1 regime)
            n_regimes_from_data = len(regime_stats)


    # Add regime probabilities if available and aligned
    if regime_probs_aligned is not None and n_regimes_from_data > 0:
        # regime_probs_aligned should be a list of lists/arrays, e.g., [[p0,p1,p2], [p0,p1,p2], ...]
        # And each inner list/array should have length n_regimes_from_data
        if regime_probs_aligned and len(regime_probs_aligned[0]) == n_regimes_from_data:
            for i in range(n_regimes_from_data):
                try:
                    df[f'Regime_{i}_Prob'] = [probs[i] for probs in regime_probs_aligned]
                except IndexError:
                    print(f"Warning Dashboard: IndexError when accessing probability for regime {i}. Prob array might be malformed.")
                    df[f'Regime_{i}_Prob'] = 0 # Fill with 0 or np.nan
        else:
            print(f"Warning Dashboard: regime_probs_aligned structure mismatch. Expected {n_regimes_from_data} probabilities per step. Got {len(regime_probs_aligned[0]) if regime_probs_aligned else 'N/A'}")

    
    # Create color palette
    colors = _create_color_palette(n_regimes_from_data)
    
    # To this:
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Price and Regimes', 'Regime Probabilities',
            'Return Distribution by Regime', 'Regime Distribution',
            'Cumulative Returns by Regime', 'Regime Transitions'
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": False}], # Price, Probs
            [{}, {"type": "domain"}], # Return Dist, Regime Dist (pie chart needs "domain" type)
            [{}, {}]  # Cum Returns, Transitions
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # --- Plotting (using df and n_regimes_from_data) ---

    # 1. Price and Regime Plot
    if not df.empty:
        # Plot Price on secondary Y-axis
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=1)
            ),
            row=1, col=1, secondary_y=True # Price on secondary_y
        )

        # Add regime background as colored rectangles
        # Create a continuous series of regime changes for highlighting
        changes = []
        if not df['Regime'].empty:
            current_regime_val = df['Regime'].iloc[0]
            current_start_idx = 0
            
            for i, r_val in enumerate(df['Regime']):
                if r_val != current_regime_val:
                    if current_regime_val is not None: # Only add if current_regime_val was valid
                         changes.append((current_start_idx, i - 1, current_regime_val))
                    current_regime_val = r_val
                    current_start_idx = i
            if current_regime_val is not None: # Add the last regime change if valid
                changes.append((current_start_idx, len(df['Regime']) - 1, current_regime_val))
        
        plotted_regime_legends = set()
        for start_idx, end_idx, r_val in changes:
            if r_val is None or r_val >= n_regimes_from_data or not colors: # Skip if regime is None or out of bounds for colors
                continue
            
            regime_desc = next((stat['description'] for stat in regime_stats if stat['regime'] == r_val), f"Regime {r_val}")
            legend_name = f'Regime {r_val}: {regime_desc}'
            show_legend_flag = legend_name not in plotted_regime_legends
            if show_legend_flag:
                plotted_regime_legends.add(legend_name)

            # Using shapes for background to avoid interfering with y-axes of price
            fig.add_shape(
                type="rect",
                xref="x1", yref="paper", # yref paper spans the whole y-axis of the subplot
                x0=df['Date'].iloc[start_idx], 
                y0=0, 
                x1=df['Date'].iloc[end_idx], 
                y1=1,
                fillcolor=colors[r_val % len(colors)], # Modulo for safety
                opacity=0.2, # Reduced opacity
                layer="below", 
                line_width=0,
                name=legend_name, # Name for potential custom legend (shapes don't auto-legend well)
            )
            # Add a dummy scatter trace for the legend of regime backgrounds
            if show_legend_flag:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None], # No data points
                        mode='markers',
                        marker=dict(color=colors[r_val % len(colors)], size=10, opacity=0.3),
                        name=legend_name,
                        showlegend=True
                    ), row=1, col=1, secondary_y=False # Attach to primary y-axis for legend grouping
                )
    else:
        fig.add_annotation(text="No data for Price/Regime Plot", row=1, col=1)


    # 2. Regime Probabilities Plot
    if f'Regime_0_Prob' in df.columns and n_regimes_from_data > 0:
        for i in range(n_regimes_from_data):
            prob_col_name = f'Regime_{i}_Prob'
            if prob_col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df[prob_col_name],
                        mode='lines',
                        stackgroup='one', # Creates stacked area chart
                        name=f'Prob R{i}',
                        line=dict(width=0.5), # Thinner lines for stacked areas
                        fillcolor=colors[i % len(colors)] if colors else None,
                        legendgroup="probs", # Group legends
                        showlegend=True
                    ),
                    row=1, col=2
                )
    else:
        fig.add_annotation(text="Regime Probabilities not available", row=1, col=2)

    
    # 3. Return Distribution by Regime
    if 'Return' in df.columns and n_regimes_from_data > 0:
        for r in range(n_regimes_from_data):
            # regime_desc = next((stat['description'] for stat in regime_stats if stat['regime'] == r), f"Regime {r}")
            regime_returns = df[df['Regime'] == r]['Return'].dropna()
            
            if len(regime_returns) > 1: # Need more than 1 point for a meaningful histogram
                fig.add_trace(
                    go.Histogram(
                        x=regime_returns,
                        name=f'Returns R{r}',
                        opacity=0.7,
                        marker_color=colors[r % len(colors)] if colors else None,
                        histnorm='probability density',
                        nbinsx=30,
                        legendgroup="returns_dist",
                        showlegend=True
                    ),
                    row=2, col=1
                )
    else:
        fig.add_annotation(text="No data for Return Distribution", row=2, col=1)

    # 4. Regime Distribution (Pie Chart)
    if not df['Regime'].empty and n_regimes_from_data > 0:
        # Count days in each regime from the aligned DataFrame
        regime_counts_df = df['Regime'].value_counts().sort_index()
        
        pie_labels = []
        pie_values = []
        pie_colors = []

        for r in range(n_regimes_from_data):
            regime_desc = next((stat['description'] for stat in regime_stats if stat['regime'] == r), f"Regime {r}")
            pie_labels.append(f'R{r}: {regime_desc}')
            pie_values.append(regime_counts_df.get(r, 0)) # Get count, or 0 if regime not present
            pie_colors.append(colors[r % len(colors)] if colors else None)
        
        if sum(pie_values) > 0: # Only plot if there are counts
            fig.add_trace(
                go.Pie(
                    labels=pie_labels,
                    values=pie_values,
                    textinfo='percent+label',
                    insidetextorientation='radial',
                    marker=dict(colors=pie_colors),
                    hole=0.3,
                    sort=False # Keep order of regimes
                ),
                row=2, col=2
            )
        else:
            fig.add_annotation(text="No data for Regime Distribution Pie", row=2, col=2)
    else:
        fig.add_annotation(text="No data for Regime Distribution Pie", row=2, col=2)
    
    # 5. Cumulative Returns by Regime
    # This plot can be misleading if regimes are short-lived or returns are not always positive in a regime.
    # It shows what $1 invested ONLY during periods of that regime would yield.
    if 'Return' in df.columns and n_regimes_from_data > 0:
        for r in range(n_regimes_from_data):
            # Create a mask for the current regime
            regime_active_mask = (df['Regime'] == r)
            
            # Calculate returns as if only trading during this regime, 0 return otherwise
            # For cumulative product, we need (1 + return)
            daily_factor = pd.Series(np.where(regime_active_mask, 1 + df['Return'], 1.0), index=df.index)
            cumulative_factor = daily_factor.cumprod()
            
            # Cumulative return is (cumulative_factor - 1)
            cumulative_return_for_regime = cumulative_factor - 1
            
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=cumulative_return_for_regime,
                    mode='lines',
                    name=f'Cum.Ret. R{r}',
                    line=dict(color=colors[r % len(colors)] if colors else None, width=2),
                    legendgroup="cum_returns",
                    showlegend=True
                ),
                row=3, col=1
            )
    else:
        fig.add_annotation(text="No data for Cumulative Returns", row=3, col=1)

    
    # 6. Transition Heatmap
    if not df['Regime'].empty and n_regimes_from_data > 1: # Need at least 2 regimes for transitions
        transitions = np.zeros((n_regimes_from_data, n_regimes_from_data))
        # Use df['Regime'] which is aligned and cleaned
        regime_sequence = df['Regime'].astype(int).values # Ensure integer type

        for i in range(len(regime_sequence) - 1):
            from_r = regime_sequence[i]
            to_r = regime_sequence[i + 1]
            if 0 <= from_r < n_regimes_from_data and 0 <= to_r < n_regimes_from_data: # Bounds check
                 transitions[from_r, to_r] += 1
        
        row_sums = transitions.sum(axis=1, keepdims=True)
        # Handle division by zero for regimes that are never transitioned from
        transition_probs_matrix = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)
        
        fig.add_trace(
            go.Heatmap(
                z=transition_probs_matrix,
                x=[f'To R{i}' for i in range(n_regimes_from_data)],
                y=[f'From R{i}' for i in range(n_regimes_from_data)],
                colorscale='Blues',
                showscale=True,
                text=np.round(transition_probs_matrix, 2),
                texttemplate="%{text}", # Show text on cells
                zmin=0, zmax=1,
                colorbar=dict(title="Prob.", thickness=10)
            ),
            row=3, col=2
        )
    else:
        fig.add_annotation(text="Not enough regimes for Transition Heatmap", row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title_text=f"Market Regime Analysis Dashboard ({results.get('ticker', 'N/A')})",
        height=1200, # Increased height for more subplots
        width=1400, # Increased width
        template="plotly_white",
        legend=dict(
            traceorder='normal', # Keep order of traces
            orientation="h",
            yanchor="bottom",
            y=1.01, # Position legend above plots
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=100, b=50) # Adjust margins
    )
    
    # Update axes titles consistently
    fig.update_yaxes(title_text="Price", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="", secondary_y=False, row=1, col=1) # No title for primary y where shapes are
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=2, range=[0,1]) # Probabilities 0-1
    
    fig.update_xaxes(title_text="Log Return", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    
    # Pie chart (subplot 2,2) doesn't need axis titles
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Return Factor", row=3, col=1) # Factor (1 = breakeven)
    
    # Heatmap (subplot 3,2) uses x/y labels from data
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dashboard_path = os.path.join(output_dir, filename)
    try:
        pio.write_html(fig, dashboard_path, auto_open=False)
        print(f"Interactive dashboard saved to {dashboard_path}")
    except Exception as e:
        print(f"Error saving dashboard to HTML: {e}")
        return None # Indicate failure
    
    return dashboard_path