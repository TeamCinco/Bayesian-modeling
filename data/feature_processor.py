import math

def normalize_features(features):
    """
    Normalize features to have 0 mean and unit variance
    
    Parameters:
    -----------
    features : dict
        Dictionary of feature arrays
        
    Returns:
    --------
    normalized : dict
        Dictionary of normalized feature arrays
    """
    normalized = {}
    
    for name, values in features.items():
        # Calculate mean and std
        valid_values = [v for v in values if v is not None and not math.isnan(v)]
        if not valid_values:
            normalized[name] = [0.0] * len(values)
            continue
            
        mean = sum(valid_values) / len(valid_values)
        variance = sum((v - mean) ** 2 for v in valid_values) / len(valid_values)
        std = math.sqrt(variance) if variance > 0 else 1.0
        
        # Normalize values
        normalized[name] = [(v - mean) / std if v is not None and not math.isnan(v) else 0.0 for v in values]
    
    return normalized


def combine_features(features, weights=None):
    """
    Combine multiple features into a single series
    
    Parameters:
    -----------
    features : dict
        Dictionary of feature arrays
    weights : dict
        Dictionary of weights for each feature
        
    Returns:
    --------
    combined : list
        Combined feature series
    """
    if weights is None:
        # Default equal weights
        weights = {name: 1.0 for name in features}
    
    # Ensure all features have the same length
    min_length = min(len(values) for values in features.values())
    
    # Initialize combined series
    combined = [0.0] * min_length
    
    # Compute weighted sum
    total_weight = sum(weights.values())
    for name, values in features.items():
        if name in weights and weights[name] > 0:
            weight = weights[name] / total_weight
            for i in range(min_length):
                combined[i] += values[i] * weight
    
    return combined
