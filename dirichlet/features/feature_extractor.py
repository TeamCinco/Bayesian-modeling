import math

class FeatureExtractor:
    """Extract features from OHLCV data for regime detection"""
    
    @staticmethod
    def log_returns(prices):
        """Calculate log returns from price series"""
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] <= 0 or prices[i] <= 0:
                returns.append(0.0)  # Handle invalid prices
            else:
                returns.append(math.log(prices[i] / prices[i-1]))
        return returns
    
    @staticmethod
    def rolling_volatility(returns, window=5):
        """Calculate rolling volatility (standard deviation)"""
        volatility = []
        for i in range(len(returns)):
            if i < window - 1:
                # Not enough data for full window, use available data
                window_returns = returns[:i+1]
            else:
                window_returns = returns[i-window+1:i+1]
            
            # Calculate std dev if enough data
            if len(window_returns) > 1:
                mean = sum(window_returns) / len(window_returns)
                variance = sum((r - mean) ** 2 for r in window_returns) / len(window_returns)
                volatility.append(math.sqrt(variance))
            else:
                volatility.append(0.0)
        
        return volatility
    
    @staticmethod
    def atr(high, low, close, window=14):
        """Calculate Average True Range"""
        tr = []  # True Range
        
        # Calculate True Range
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr.append(max(tr1, tr2, tr3))
        
        # Calculate ATR with Simple Moving Average
        atr = []
        for i in range(len(tr)):
            if i < window:
                # Use simple average for initial values
                atr.append(sum(tr[:i+1]) / (i+1))
            else:
                # Use simple moving average
                atr.append(sum(tr[i-window+1:i+1]) / window)
        
        # Add a 0 for the first day (no TR calculated)
        return [0.0] + atr
    
    @staticmethod
    def bollinger_width(close, window=20, num_std=2):
        """Calculate Bollinger Band width (normalized)"""
        bb_width = []
        
        for i in range(len(close)):
            if i < window - 1:
                # Not enough data, use simple estimate
                mean = sum(close[:i+1]) / (i+1)
                if i > 0:
                    std = math.sqrt(sum((p - mean) ** 2 for p in close[:i+1]) / (i+1))
                else:
                    std = 0.0
            else:
                window_prices = close[i-window+1:i+1]
                mean = sum(window_prices) / window
                std = math.sqrt(sum((p - mean) ** 2 for p in window_prices) / window)
            
            # Calculate width and normalize by mean price
            width = (2 * num_std * std)
            if mean > 0:
                bb_width.append(width / mean)  # Normalized by price level
            else:
                bb_width.append(0.0)
        
        return bb_width
    
    @staticmethod
    def momentum(close, short_window=10, long_window=20):
        """Calculate price momentum relative to moving averages"""
        momentum = []
        
        for i in range(len(close)):
            if i < short_window - 1:
                # Not enough data for short MA
                momentum.append(0.0)
            elif i < long_window - 1:
                # Only short MA available
                short_ma = sum(close[i-short_window+1:i+1]) / short_window
                momentum.append(close[i] / short_ma - 1.0)
            else:
                # Both MAs available
                short_ma = sum(close[i-short_window+1:i+1]) / short_window
                long_ma = sum(close[i-long_window+1:i+1]) / long_window
                # Ratio of short MA to long MA
                momentum.append(short_ma / long_ma - 1.0)
        
        return momentum
    
    @staticmethod
    def rsi(close, window=14):
        """Calculate Relative Strength Index"""
        rsi = []
        
        # Need at least 2 prices to calculate gains/losses
        if len(close) < 2:
            return [50.0] * len(close)
        
        # Calculate price changes
        changes = [close[i] - close[i-1] for i in range(1, len(close))]
        
        # Separate gains and losses
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        # Calculate RSI
        for i in range(len(changes)):
            if i < window:
                # Use simple averages for first window periods
                avg_gain = sum(gains[:i+1]) / (i+1)
                avg_loss = sum(losses[:i+1]) / (i+1)
            else:
                # Use wilder's smoothing
                avg_gain = (gains[i] + (window - 1) * avg_gain) / window
                avg_loss = (losses[i] + (window - 1) * avg_loss) / window
            
            if avg_loss == 0:
                rsi.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
        
        # Add a value for the first day
        return [50.0] + rsi
    
    @staticmethod
    def extract_features(ohlcv_data, config=None):
        """
        Extract all features from OHLCV data
        
        Parameters:
        -----------
        ohlcv_data : dict
            Dictionary with 'open', 'high', 'low', 'close', 'volume' keys
        config : dict
            Feature configuration parameters
            
        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        # Default config
        if config is None:
            config = {
                'volatility_window': 5,
                'atr_window': 14,
                'bb_window': 20,
                'momentum_short': 10,
                'momentum_long': 20,
                'rsi_window': 14
            }
        
        # Extract individual features
        features = {}
        features['log_returns'] = FeatureExtractor.log_returns(ohlcv_data['close'])
        features['volatility'] = FeatureExtractor.rolling_volatility(
            features['log_returns'], window=config['volatility_window']
        )
        features['atr'] = FeatureExtractor.atr(
            ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'], 
            window=config['atr_window']
        )
        features['bb_width'] = FeatureExtractor.bollinger_width(
            ohlcv_data['close'], window=config['bb_window']
        )
        features['momentum'] = FeatureExtractor.momentum(
            ohlcv_data['close'], short_window=config['momentum_short'], long_window=config['momentum_long']
        )
        features['rsi'] = FeatureExtractor.rsi(
            ohlcv_data['close'], window=config['rsi_window']
        )
        
        return features
