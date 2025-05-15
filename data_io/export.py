"""
Export functions for market regime analysis.

This module provides functions to export regime data to various formats
such as JSON and CSV files for further analysis or integration with
other systems.
"""

import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime, date  # Import date explicitly for isinstance check


def export_regime_data_to_json(results, output_path, indent=2):
    """
    Export regime analysis results to JSON format
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary containing regimes, statistics, etc.
    output_path : str
        Path to save the JSON file
    indent : int, optional
        Indentation level for the JSON file
        
    Returns:
    --------
    output_path : str
        Path to the created JSON file
    """
    # Extract data
    dates = results['dates']
    regimes = results['regimes']
    regime_stats = results['regime_stats']
    regime_probs = results.get('regime_probs', None)
    
    # Debug information to identify length mismatches
    print(f"Debug - Lengths: dates={len(dates)}, regimes={len(regimes)}")
    if regime_probs:
        print(f"Debug - regime_probs length: {len(regime_probs)}")
    
    # Create output dictionary
    output_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_regimes': max(regimes) + 1 if regimes else 0,  # Handle empty regimes list
        'n_observations': len(dates),
        'data_range': {
            'start_date': dates[0] if dates else None,
            'end_date': dates[-1] if dates else None
        },
        'current_regime': {
            'regime': regimes[-1] if regimes else None,
            'probability': regime_probs[-1][regimes[-1]] if regime_probs and regimes and regime_probs[-1] else None
        },
        'regime_statistics': regime_stats,
    }
    
    # Add time series data
    time_series = []
    if dates:  # Ensure dates list is not empty
        # Find the minimum length between dates and regimes to avoid index errors
        min_length = min(len(dates), len(regimes)) if regimes else 0
        
        for i in range(min_length):
            try:
                entry = {
                    'date': dates[i],
                    'regime': int(regimes[i])
                }
                
                # Add regime probabilities if available
                if regime_probs and i < len(regime_probs) and regime_probs[i]:
                    entry['probabilities'] = {
                        f'regime_{j}': float(regime_probs[i][j]) 
                        for j in range(len(regime_probs[i]))
                    }
                    
                time_series.append(entry)
            except (IndexError, ValueError) as e:
                print(f"Warning: Issue processing entry at index {i}: {str(e)}")
                continue
    
    output_data['time_series'] = time_series
    
    # Ensure directory exists
    # Check if output_path has a directory part
    dir_name = os.path.dirname(output_path)
    if dir_name:  # Only create if there's a directory part
        os.makedirs(dir_name, exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        # Handle NumPy types and datetime objects that aren't JSON serializable
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                    return {'real': obj.real, 'imag': obj.imag}
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.void):
                    return None
                elif isinstance(obj, (datetime, date)):  # Handle datetime.datetime and datetime.date
                    return obj.isoformat()
                return json.JSONEncoder.default(self, obj)
        
        json.dump(output_data, f, cls=NumpyEncoder, indent=indent)
    
    print(f"Regime data exported to JSON: {output_path}")
    return output_path


def export_regime_data_to_csv(results, output_path, include_probabilities=True):
    """
    Export regime analysis results to CSV format
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary containing regimes, statistics, etc.
    output_path : str
        Path to save the CSV file
    include_probabilities : bool, optional
        Whether to include regime probabilities in the output
        
    Returns:
    --------
    output_path : str
        Path to the created CSV file
    """
    # Extract data
    dates = results['dates']
    regimes = results['regimes']
    regime_probs = results.get('regime_probs', None)
    features = results.get('features', {})
    ohlcv = results.get('ohlcv', {})
    
    # Print diagnostics
    print(f"CSV Export - Lengths: dates={len(dates)}, regimes={len(regimes)}")
    
    # Find the common length for all arrays to avoid DataFrame creation errors
    base_length = min(len(dates), len(regimes))
    print(f"Using common length: {base_length}")
    
    # Truncate arrays to common length
    dates_truncated = dates[:base_length]
    regimes_truncated = regimes[:base_length]
    
    # Create DataFrame
    data = {'Date': dates_truncated, 'Regime': regimes_truncated}
    
    # Add OHLCV data if available
    for key in ['open', 'high', 'low', 'close', 'volume']:
        if key in ohlcv and len(ohlcv[key]) > 0:
            data[key.capitalize()] = ohlcv[key][:base_length]  # Truncate to common length
    
    # Add features if available
    for key, values in features.items():
        if isinstance(values, list) and len(values) > 0:
            # Skip the combined feature which is usually an internal representation
            if key != 'combined':
                # Truncate to common length
                truncated_values = values[:base_length]
                data[key.replace('_', ' ').title()] = truncated_values
    
    # Add regime probabilities if available and requested
    if include_probabilities and regime_probs and len(regime_probs) > 0:
        regime_probs_truncated = regime_probs[:base_length]
        if regime_probs_truncated:
            n_regimes = len(regime_probs_truncated[0]) if regime_probs_truncated[0] else 0
            for i in range(n_regimes):
                try:
                    probs_column = [probs[i] for probs in regime_probs_truncated]
                    if len(probs_column) == base_length:
                        data[f'Regime_{i}_Probability'] = probs_column
                except (IndexError, TypeError) as e:
                    print(f"Warning: Error processing probabilities for regime {i}: {e}")
                    continue

    # Create DataFrame
    try:
        df = pd.DataFrame(data)
        
        # Ensure directory exists
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Regime data exported to CSV: {output_path}")
        return output_path
        
    except ValueError as e:
        print(f"Error creating DataFrame for CSV export: {e}")
        print("Column lengths in data dictionary:")
        for key, value in data.items():
            print(f"  {key}: {len(value) if hasattr(value, '__len__') else 'not a sequence'}")
        raise RuntimeError(f"Failed to create DataFrame: {e}")

def export_transition_matrix(results, output_path, format='csv'):
    """
    Export the regime transition probability matrix
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary containing the HMM model
    output_path : str
        Path to save the transition matrix
    format : str, optional
        Format to save the matrix ('csv' or 'json')
        
    Returns:
    --------
    output_path : str
        Path to the created file
    """
    # Extract the HMM model
    hmm = results.get('hmm_model')
    if hmm is None:
        # Try to get it from the 'hmm' key if 'hmm_model' is not present
        hmm = results.get('hmm') 
        if hmm is None:
            print("Warning: No HMM model found in results under 'hmm_model' or 'hmm'. Cannot export transition matrix.")
            return None
    
    # Get transition matrix
    # Check if hmm object has 'transmat_' or 'trans_mat'
    if hasattr(hmm, 'transmat_') and hmm.transmat_ is not None:
        transition_matrix = hmm.transmat_
    elif hasattr(hmm, 'trans_mat') and hmm.trans_mat is not None:
        transition_matrix = hmm.trans_mat
    else:
        print("Warning: HMM model does not have a 'transmat_' or 'trans_mat' attribute, or it is None. Cannot export transition matrix.")
        return None

    # Print debug info about transition_matrix type
    print(f"Debug - Transition matrix type: {type(transition_matrix)}")
    
    # Convert transition matrix to numpy array if it's a list
    if isinstance(transition_matrix, list):
        try:
            # Get dimensions from the first row of the list
            if transition_matrix and isinstance(transition_matrix[0], list):
                n_regimes = len(transition_matrix)
                print(f"Using n_regimes={n_regimes} from list length")
                
                # Convert to numpy array for easier handling
                transition_matrix_np = np.array(transition_matrix)
                print(f"Successfully converted transition matrix from list to numpy array of shape {transition_matrix_np.shape}")
            else:
                print("Warning: Transition matrix is not in expected format (list of lists).")
                n_regimes = len(transition_matrix)
                # Try to reshape if it's a flat list
                transition_matrix_np = np.array(transition_matrix).reshape(n_regimes, n_regimes)
                print(f"Reshaped flat list to {n_regimes}x{n_regimes} matrix")
        except Exception as e:
            print(f"Error converting transition matrix: {str(e)}")
            return None
    elif isinstance(transition_matrix, np.ndarray):
        # Already a numpy array
        transition_matrix_np = transition_matrix
        n_regimes = transition_matrix_np.shape[0]
        print(f"Transition matrix is already a numpy array of shape {transition_matrix_np.shape}")
    else:
        print(f"Unexpected transition matrix type: {type(transition_matrix)}")
        return None
    
    # Ensure directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    if format.lower() == 'csv':
        # Create DataFrame with row and column labels
        df = pd.DataFrame(
            transition_matrix_np,
            index=[f'From Regime {i}' for i in range(n_regimes)],
            columns=[f'To Regime {i}' for i in range(n_regimes)]
        )
        
        # Export to CSV
        df.to_csv(output_path)
        
    elif format.lower() == 'json':
        # Create a dictionary
        matrix_dict = {
            'n_regimes': n_regimes,
            'description': 'Regime transition probability matrix',
            'matrix': transition_matrix_np.tolist(),
            'row_labels': [f'From Regime {i}' for i in range(n_regimes)],
            'column_labels': [f'To Regime {i}' for i in range(n_regimes)]
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            # Use the NumpyEncoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                        np.int16, np.int32, np.int64, np.uint8,
                                        np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                        return {'real': obj.real, 'imag': obj.imag}
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.void):
                        return None
                    elif isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    return json.JSONEncoder.default(self, obj)
            
            json.dump(matrix_dict, f, cls=NumpyEncoder, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")
    
    print(f"Transition matrix exported to {format.upper()}: {output_path}")
    return output_path