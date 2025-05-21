# volatility_calculator.py
"""
Contains the logic for calculating historical volatility.
"""
import pandas as pd
import numpy as np

def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculates rolling window standard deviation of log returns, annualized.
    """
    # Ensure 'close' column exists and has data
    if 'close' not in df.columns or df['close'].isnull().all():
        return pd.Series(dtype=float, index=df.index if df is not None else None)
    
    # Work with positive prices only for log returns
    prices = df['close'][df['close'] > 0].copy() 
    if prices.empty:
        return pd.Series(dtype=float, index=df.index if df is not None else None)
        
    shifted_prices = prices.shift(1)
    # Calculate log returns only where both current and shifted prices are positive
    valid_for_log_return = (prices > 0) & (shifted_prices > 0) 
    log_returns = pd.Series(np.nan, index=df.index, dtype=float) # Initialize with NaNs on original index
    
    if valid_for_log_return.any():
        # Compute log returns for the valid subset
        computed_log_returns_subset = np.log(prices[valid_for_log_return] / 
                                             shifted_prices[valid_for_log_return])
        # Assign back to the correctly indexed series
        log_returns.loc[computed_log_returns_subset.index] = computed_log_returns_subset

    # If no valid log returns could be calculated (e.g., only one price point after filtering)
    if log_returns.dropna().empty:
        return pd.Series(dtype=float, index=df.index if df is not None else None)
        
    # Adjust min_periods for rolling calculation
    available_log_returns_count = len(log_returns.dropna())
    min_p = max(1, window // 2) # Ensure min_periods is at least 1
    if available_log_returns_count < min_p : # If not enough data for even min_periods
        min_p = max(1,available_log_returns_count) # Use what's available, down to 1

    # Calculate rolling standard deviation and annualize
    volatility = log_returns.rolling(window=window, min_periods=min_p).std() * np.sqrt(252)
    return volatility