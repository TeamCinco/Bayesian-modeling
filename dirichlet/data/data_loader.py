import csv
from datetime import datetime

def load_csv_data(file_path):
    """
    Load OHLCV data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing OHLCV data
        
    Returns:
    --------
    dates : list
        List of datetime.date objects
    ohlcv : dict
        Dictionary with 'open', 'high', 'low', 'close', 'volume' keys
    """
    dates = []
    ohlcv = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle the datetime format in your files
            date_str = row.get('datetime', '')
            if date_str:
                try:
                    # Parse datetime and extract date portion
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    dates.append(date_obj.date())
                except ValueError:
                    # Try alternative format if first one fails
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        dates.append(date_obj.date())
                    except ValueError:
                        continue  # Skip rows with invalid dates
            
            try:
                ohlcv['open'].append(float(row['open']))
                ohlcv['high'].append(float(row['high']))
                ohlcv['low'].append(float(row['low']))
                ohlcv['close'].append(float(row['close']))
                ohlcv['volume'].append(float(row['volume']))
            except (ValueError, KeyError) as e:
                # Handle missing data by using previous values or default
                if ohlcv['open']:  # If we have previous values
                    ohlcv['open'].append(ohlcv['open'][-1])
                    ohlcv['high'].append(ohlcv['high'][-1])
                    ohlcv['low'].append(ohlcv['low'][-1])
                    ohlcv['close'].append(ohlcv['close'][-1])
                    ohlcv['volume'].append(0.0)
                else:  # No previous values
                    ohlcv['open'].append(0.0)
                    ohlcv['high'].append(0.0)
                    ohlcv['low'].append(0.0)
                    ohlcv['close'].append(0.0)
                    ohlcv['volume'].append(0.0)
    
    return dates, ohlcv
