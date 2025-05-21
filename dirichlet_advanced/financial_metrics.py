# financial_metrics.py
"""
Functions for calculating financial metrics, like regime-based return statistics.
"""
import pandas as pd
import numpy as np

def calculate_regime_return_statistics(data_df: pd.DataFrame, 
                                     regimes_series: pd.Series, 
                                     actual_num_regimes: int) -> list:
    """Calculate detailed return statistics for each volatility regime."""
    
    results_list = []
    # Default entry for regimes with no data
    def default_stats(regime_idx):
        return {
            'Regime': regime_idx, 'N_Days': 0, 'Proportion_Of_Time': 0,
            'Mean_Daily_Log_Return': np.nan, 'StdDev_Daily_Log_Return': np.nan,
            'Annualized_Sharpe_Ratio': np.nan, 'Skewness_Daily_Log_Return': np.nan,
            'Excess_Kurtosis': np.nan, 'Total_Compounded_Return': np.nan,
            'Max_Drawdown': np.nan, 'Avg_Drawdown': np.nan,
            'Avg_5Day_Forward_Return': np.nan
        }

    if 'close' not in data_df.columns or actual_num_regimes == 0 or regimes_series.dropna().empty:
        return [default_stats(i) for i in range(actual_num_regimes)]

    prices = data_df['close']
    # Calculate log returns more robustly
    log_returns = pd.Series(np.nan, index=prices.index, dtype=float)
    valid_ret_idx = (prices > 0) & (prices.shift(1) > 0)
    log_returns[valid_ret_idx] = np.log(prices[valid_ret_idx] / prices.shift(1)[valid_ret_idx])
    
    # Combine with regimes, dropping NaNs from returns or regimes
    stats_df = pd.DataFrame({'LogReturn': log_returns, 'Regime': regimes_series}).dropna()
    total_valid_days_in_stats = len(stats_df)

    for i in range(actual_num_regimes):
        regime_specific_returns = stats_df[stats_df['Regime'] == i]['LogReturn']
        n_days = len(regime_specific_returns)

        if n_days > 1: # Need at least 2 days for std dev and other stats
            mean_ret = regime_specific_returns.mean()
            std_ret = regime_specific_returns.std()
            # Sharpe: annualize mean return (daily_mean * 252), annualize std dev (daily_std * sqrt(252))
            sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 1e-9 else \
                     (0.0 if abs(mean_ret * 252) < 1e-9 else np.nan)

            skew_ret = regime_specific_returns.skew() if n_days > 2 else np.nan
            kurt_ret = regime_specific_returns.kurtosis() if n_days > 3 else np.nan # Fisher (excess) kurtosis
            total_ret_comp = np.exp(regime_specific_returns.sum()) - 1 # Compounded over regime duration

            # Max Drawdown calculation
            cumulative_log_returns = regime_specific_returns.cumsum()
            # Using portfolio value for drawdown: Add 1 before cumprod if using arithmetic returns, or exp(cumsum) for log
            portfolio_value = np.exp(cumulative_log_returns)
            running_max_val = portfolio_value.cummax()
            drawdown = (portfolio_value - running_max_val) / running_max_val # Relative drawdown
            max_drawdown = drawdown.min() if not drawdown.empty else np.nan
            avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
            
            # 5-day forward returns calculation
            forward_returns_list = []
            for date_idx in regime_specific_returns.index:
                # Find current date's position in the original full data_df.index
                loc_in_fulldf = data_df.index.get_loc(date_idx)
                # Ensure target forward date is within bounds of full data_df
                if loc_in_fulldf + 5 < len(data_df.index):
                    forward_date = data_df.index[loc_in_fulldf + 5]
                    # Ensure prices exist on both dates
                    if date_idx in prices.index and forward_date in prices.index and \
                       prices[date_idx] > 0 and prices[forward_date] > 0:
                        fwd_ret = np.log(prices[forward_date] / prices[date_idx])
                        forward_returns_list.append(fwd_ret)
            avg_5d_fwd_ret = np.mean(forward_returns_list) if forward_returns_list else np.nan
            
            results_list.append({
                'Regime': i, 'N_Days': n_days, 
                'Proportion_Of_Time': n_days / total_valid_days_in_stats if total_valid_days_in_stats > 0 else 0,
                'Mean_Daily_Log_Return': mean_ret, 'StdDev_Daily_Log_Return': std_ret,
                'Annualized_Sharpe_Ratio': sharpe, 'Skewness_Daily_Log_Return': skew_ret,
                'Excess_Kurtosis': kurt_ret, 'Total_Compounded_Return': total_ret_comp,
                'Max_Drawdown': max_drawdown, 'Avg_Drawdown': avg_drawdown,
                'Avg_5Day_Forward_Return': avg_5d_fwd_ret
            })
        else:
            results_list.append(default_stats(i)) # Use default for regimes with <2 days
            if n_days == 1 and 'Regime' in results_list[-1]: # If 1 day, can report proportion
                 results_list[-1]['Proportion_Of_Time'] = n_days / total_valid_days_in_stats if total_valid_days_in_stats > 0 else 0

    return results_list