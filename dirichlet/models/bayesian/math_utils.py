import math

def gaussian_pdf(x, mu, sigma):
    """Compute Gaussian probability density function"""
    if sigma <= 0:
        return 0.0
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    try:
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)
    except OverflowError:
        # Handle numerical underflow
        if exponent < -700:  # Approximate lower bound for exp in Python
            return 1e-300  # Small positive number instead of zero
        return 0.0

def log_sum_exp(a, b):
    """Numerically stable implementation of log(exp(a) + exp(b))"""
    if a == -float('inf'):
        return b
    if b == -float('inf'):
        return a
    if a > b:
        return a + math.log(1 + math.exp(b - a))
    else:
        return b + math.log(1 + math.exp(a - b))
