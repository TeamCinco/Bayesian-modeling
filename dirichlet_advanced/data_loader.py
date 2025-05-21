# data_loader.py
"""
Handles loading and initial standardization of stock CSV data.
"""
import pandas as pd
import numpy as np
import glob
import os

def load_stock_data(folder_path: str) -> dict:
    """
    Load all CSV files from a folder into a dictionary of DataFrames,
    standardizing OHLC column names and setting a DatetimeIndex.
    """
    all_data = {}
    name_map = {
        'open': ['open', 'o', 'first'],
        'high': ['high', 'h', 'max'],
        'low': ['low', 'l', 'min'],
        'close': ['close', 'c', 'last', 'price']
    }
    required_cols_std = ['open', 'high', 'low', 'close']

    for file_path in glob.glob(os.path.join(folder_path, "*.csv")):
        symbol = os.path.basename(file_path).split('.')[0]
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()

            rename_dict = {}
            found_cols_orig_names = {} # Store original names that were mapped
            for std_name, potential_names in name_map.items():
                for pot_name in potential_names:
                    if pot_name in df.columns:
                        if std_name not in found_cols_orig_names: # Take first match
                           rename_dict[pot_name] = std_name
                           found_cols_orig_names[std_name] = pot_name
                        break
            df.rename(columns=rename_dict, inplace=True)

            for col_std in required_cols_std: # Ensure standard OHLC cols exist
                if col_std not in df.columns: df[col_std] = np.nan
                df[col_std] = pd.to_numeric(df[col_std], errors='coerce')
            
            if 'close' not in df.columns or df['close'].isnull().all(): continue
            df.dropna(subset=['close'], inplace=True)
            if df.empty: continue

            date_cols = [col for col in df.columns if any(s in col.lower() for s in ['date', 'time', 'timestamp'])]
            if date_cols:
                date_col_name = date_cols[0]
                try:
                    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
                    df.dropna(subset=[date_col_name], inplace=True)
                    if df.empty: continue
                    df.set_index(date_col_name, inplace=True)
                    df.sort_index(inplace=True)
                except Exception as e:
                    print(f"Could not parse/set dates for {symbol} ({date_col_name}): {e}. Skipping.")
                    continue
            elif isinstance(df.index, pd.DatetimeIndex):
                 if not df.index.is_monotonic_increasing: df.sort_index(inplace=True)
            else:
                print(f"Warning: No date column or DatetimeIndex for {symbol}. Skipping.")
                continue
            
            # Keep only standardized OHLC and any other original columns that were successfully mapped
            cols_to_keep = required_cols_std + [col for col in df.columns if col not in required_cols_std and col in rename_dict.values()]
            # Ensure no duplicates and all required_cols_std are present if they were in original df
            final_cols = []
            seen_cols = set()
            for col in cols_to_keep:
                if col not in seen_cols:
                    final_cols.append(col)
                    seen_cols.add(col)
            
            # Add other non-OHLC columns that were not part of name_map
            other_original_cols = [c for c in df.columns if c not in final_cols and c not in rename_dict.keys() and c not in required_cols_std]
            final_cols.extend(other_original_cols)

            all_data[symbol] = df[final_cols].copy()

        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")
    return all_data