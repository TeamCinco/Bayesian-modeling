# data_utils.py
"""
Utility functions for data manipulation, primarily date filtering.
"""
import pandas as pd
import numpy as np # Not directly used but often goes with pandas

def filter_data_by_date(data: pd.DataFrame, 
                        start_date_str: str = '2010-01-01', 
                        end_date_str: str = None) -> pd.DataFrame:
    """
    Filters a DataFrame by date, assuming a DatetimeIndex.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        # print(f"Data for a stock does not have a DatetimeIndex. Skipping date filtering.")
        return data 
    
    if not data.index.is_monotonic_increasing: 
        data = data.sort_index()

    start_ts = pd.Timestamp(start_date_str)
    end_ts = pd.Timestamp(end_date_str) if end_date_str else pd.Timestamp.now().normalize() 

    # Handle timezone awareness
    if data.index.tz is not None:
        start_ts = start_ts.tz_localize(data.index.tz) if start_ts.tzinfo is None \
            else start_ts.tz_convert(data.index.tz)
        end_ts = end_ts.tz_localize(data.index.tz) if end_ts.tzinfo is None \
            else end_ts.tz_convert(data.index.tz)
    else: # data.index is tz-naive
        start_ts = start_ts.tz_localize(None) if start_ts.tzinfo is not None else start_ts
        end_ts = end_ts.tz_localize(None) if end_ts.tzinfo is not None else end_ts
            
    filtered_data = data.loc[(data.index >= start_ts) & (data.index <= end_ts)].copy()
    return filtered_data