# volatility_models.py
"""
Advanced volatility models: GARCH, Stochastic Volatility, EWMA, Neural Network (PyTorch), Hybrid.
"""
import pandas as pd
import numpy as np
from volatility_calculator import calculate_volatility # For fallbacks
import config # For default parameters

# GARCH
def calculate_garch_volatility(df, p=config.DEFAULT_GARCH_P, q=config.DEFAULT_GARCH_Q,
                               fallback_vol_func=calculate_volatility,
                               fallback_window=config.DEFAULT_VOLATILITY_WINDOW):
    try:
        from arch import arch_model
    except ImportError:
        print("arch library not found. GARCH model unavailable. Falling back.")
        return fallback_vol_func(df, window=fallback_window)

    if 'close' not in df.columns or df['close'].isnull().all():
        return pd.Series(dtype=float, index=df.index)

    prices = df['close'][df['close'] > 0].copy()
    returns = np.log(prices / prices.shift(1)).dropna() * 100 # Scale for GARCH

    if len(returns) < 30: # GARCH needs sufficient data
        return fallback_vol_func(df, window=fallback_window)
    
    model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
    try:
        results = model.fit(disp='off', show_warning=False) # Suppress verbose output
        conditional_vol_daily_pct = results.conditional_volatility
        conditional_vol = (conditional_vol_daily_pct / 100) * np.sqrt(252) # Rescale and annualize
        volatility = pd.Series(np.nan, index=df.index)
        volatility.loc[returns.index] = conditional_vol
        return volatility.ffill()
    except Exception: # Broad exception for GARCH fitting issues
        return fallback_vol_func(df, window=fallback_window)

# Stochastic Volatility (using PyMC3)
def calculate_sv_volatility(df, fallback_vol_func=calculate_volatility,
                            fallback_window=config.DEFAULT_VOLATILITY_WINDOW):
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError:
        print("pymc not found or import error. SV model unavailable. Falling back.")
        return fallback_vol_func(df, window=fallback_window)
    except Exception as e_runtime_import:
        print(f"Runtime import error for PyMC components: {e_runtime_import}. SV model unavailable. Falling back.")
        return fallback_vol_func(df, window=fallback_window)

    if 'close' not in df.columns or df['close'].isnull().all():
        return pd.Series(dtype=float, index=df.index)

    prices = df['close'][df['close'] > 0].copy()
    returns = prices.pct_change().dropna() * 100

    if len(returns) < 50:  # SV models need more data
        return fallback_vol_func(df, window=fallback_window)
    
    try:
        # Define coordinates for our model
        coords = {"time": returns.index}
        
        with pm.Model(coords=coords) as sv_model:
            # Model parameters
            sigma_h = pm.Exponential('sigma_h', lam=1.0/0.1)
            phi_h = pm.Uniform('phi_h', lower=-0.99, upper=0.99)
            mu_h = pm.Normal('mu_h', mu=-5, sigma=2)
            
            # Set up autoregressive volatility process
            h_init = pm.Normal('h_init', 
                              mu=mu_h, 
                              sigma=sigma_h / pt.sqrt(1 - phi_h**2 + 1e-6))
            
            h_innovations = pm.Normal('h_innovations', 
                                     mu=0, 
                                     sigma=sigma_h, 
                                     dims="time")
            
            # Define the autoregressive process for log-volatility
            h = pm.Deterministic('h', 
                               pm.math.scan(
                                   lambda h_prev, innov: mu_h * (1 - phi_h) + phi_h * h_prev + innov,
                                   sequences=[h_innovations],
                                   outputs_info=[h_init]
                               ),
                               dims="time")
            
            # Likelihood of observed returns
            returns_obs = pm.Normal('returns_obs', 
                                  mu=0, 
                                  sigma=pt.exp(h / 2.0), 
                                  observed=returns.values,
                                  dims="time")
            
            # Sample from the posterior
            idata = pm.sample(500, tune=500, chains=2, cores=1,
                             nuts_kwargs={'target_accept': 0.9},
                             progressbar=False)

        # Extract results
        h_means = idata.posterior['h'].mean(dim=["chain", "draw"]).values
        sv_daily_vol_pct = np.exp(h_means / 2.0)
        sv_daily_vol = sv_daily_vol_pct / 100.0
        annualized_sv_vol = sv_daily_vol * np.sqrt(252)
        
        volatility = pd.Series(np.nan, index=df.index)
        volatility.loc[returns.index] = annualized_sv_vol
        return volatility.ffill()
    except Exception as e:
        print(f"Error in SV model: {e}")
        return fallback_vol_func(df, window=fallback_window)
# EWMA
def calculate_ewma_volatility(df, decay=config.DEFAULT_EWMA_DECAY,
                              fallback_vol_func=calculate_volatility,
                              fallback_window=config.DEFAULT_VOLATILITY_WINDOW):
    if 'close' not in df.columns or df['close'].isnull().all():
        return pd.Series(dtype=float, index=df.index)

    prices = df['close'][df['close'] > 0].copy()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if log_returns.empty or len(log_returns) < 2:
        return fallback_vol_func(df, window=fallback_window)

    ewma_var_manual = pd.Series(index=log_returns.index, dtype=float)
    # Initialize with sample variance of first few observations (e.g., min of 20 or available)
    initial_var_period = min(fallback_window, len(log_returns)) # Use fallback_window for consistency
    if initial_var_period > 1 :
        ewma_var_manual.iloc[0] = np.var(log_returns.iloc[:initial_var_period], ddof=0) # Population variance
    elif not log_returns.empty:
         ewma_var_manual.iloc[0] = log_returns.iloc[0]**2
    else: # Should not happen if log_returns.empty is checked
        return fallback_vol_func(df, window=fallback_window)

    if pd.isna(ewma_var_manual.iloc[0]): # If var is still NaN (e.g. single point for init)
        ewma_var_manual.iloc[0] = log_returns.iloc[0]**2 if not log_returns.empty else 0.0

    for i in range(1, len(log_returns)):
        prev_var = ewma_var_manual.iloc[i-1]
        # Ensure prev_var is not NaN; if so, might re-initialize or use current squared return
        if pd.isna(prev_var): prev_var = log_returns.iloc[i-1]**2 if i>0 else 0.0

        ewma_var_manual.iloc[i] = (1 - decay) * (log_returns.iloc[i-1]**2) + decay * prev_var

    ewma_vol = np.sqrt(ewma_var_manual) * np.sqrt(252)
    
    volatility = pd.Series(np.nan, index=df.index)
    volatility.loc[ewma_vol.index] = ewma_vol
    return volatility.ffill()

# Neural Network Volatility (PyTorch Implementation)
def calculate_nn_volatility(df, lookback=config.DEFAULT_NN_LOOKBACK,
                            fallback_vol_func=calculate_volatility,
                            fallback_window=config.DEFAULT_VOLATILITY_WINDOW):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("PyTorch or sklearn not found. NN model unavailable. Falling back.")
        return fallback_vol_func(df, window=fallback_window)

    if 'close' not in df.columns or df['close'].isnull().all():
        return pd.Series(dtype=float, index=df.index)

    prices = df['close'][df['close'] > 0].copy()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Need enough data for lookback, training, and to form at least one sequence
    if len(log_returns) < lookback + 20: # Arbitrary: lookback for sequence + 20 for training
        return fallback_vol_func(df, window=fallback_window)

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    squared_log_returns = log_returns**2
    
    X_data_np, y_data_np = [], []
    for i in range(lookback, len(squared_log_returns)):
        X_data_np.append(squared_log_returns.iloc[i-lookback:i].values)
        y_data_np.append(squared_log_returns.iloc[i])
    
    X_np = np.array(X_data_np); y_np = np.array(y_data_np).reshape(-1, 1)
    if X_np.shape[0] == 0: return fallback_vol_func(df, window=fallback_window)

    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_scaled_np = scaler_X.fit_transform(X_np)
    y_scaled_np = scaler_y.fit_transform(y_np)

    X_tensor = torch.FloatTensor(X_scaled_np).unsqueeze(2).to(device)
    y_tensor = torch.FloatTensor(y_scaled_np).to(device)

    class LSTMVolatility(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=32, output_size=1, num_layers=1, dropout=0.1): # Simpler model
            super().__init__(); self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.linear = nn.Linear(hidden_layer_size, output_size)
        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq); last_time_step_out = lstm_out[:, -1, :]
            return self.linear(last_time_step_out)

    model = LSTMVolatility().to(device) # Use simplified model
    loss_function = nn.MSELoss(); optimizer = optim.Adam(model.parameters(), lr=0.005) # Adjusted LR

    epochs = 15; batch_size = 8 # Faster training parameters
    for i in range(epochs):
        for j in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[j : j+batch_size]; y_batch = y_tensor[j : j+batch_size]
            optimizer.zero_grad(); y_pred = model(X_batch)
            single_loss = loss_function(y_pred, y_batch); single_loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad(): predicted_scaled_y_tensor = model(X_tensor)
    
    predicted_scaled_y_np = predicted_scaled_y_tensor.cpu().numpy()
    predicted_y_np = scaler_y.inverse_transform(predicted_scaled_y_np)
    
    predicted_var_values = np.maximum(predicted_y_np.flatten(), 1e-9) # Avoid zero for sqrt
    predicted_vol_values = np.sqrt(predicted_var_values) * np.sqrt(252)

    volatility = pd.Series(np.nan, index=df.index)
    volatility.loc[squared_log_returns.index[lookback:]] = predicted_vol_values
    return volatility.ffill()

# Hybrid Volatility
def calculate_hybrid_volatility(df, fallback_vol_func=calculate_volatility,
                                fallback_window=config.DEFAULT_VOLATILITY_WINDOW):
    vol_series_dict = {}
    # Always run simpler models
    vol_series_dict['simple'] = fallback_vol_func(df.copy(), window=fallback_window)
    vol_series_dict['ewma'] = calculate_ewma_volatility(df.copy(), fallback_vol_func=fallback_vol_func, fallback_window=fallback_window)
    vol_series_dict['garch'] = calculate_garch_volatility(df.copy(), fallback_vol_func=fallback_vol_func, fallback_window=fallback_window)

    # Conditionally run expensive models
    use_expensive_models = len(df) > 200 # Adjusted threshold for expensive models
    if use_expensive_models:
        vol_series_dict['sv'] = calculate_sv_volatility(df.copy(), fallback_vol_func=fallback_vol_func, fallback_window=fallback_window)
        vol_series_dict['nn'] = calculate_nn_volatility(df.copy(), fallback_vol_func=fallback_vol_func, fallback_window=fallback_window)
    else: # Fill with NaNs if not run, so DataFrame structure is consistent
        vol_series_dict['sv'] = pd.Series(np.nan, index=df.index)
        vol_series_dict['nn'] = pd.Series(np.nan, index=df.index)

    combined_vol_df = pd.DataFrame(vol_series_dict)
    # Adjusted weights, giving more to GARCH/EWMA as they are more standard
    weights = {'simple': 0.05, 'ewma': 0.30, 'garch': 0.40, 'sv': 0.10, 'nn': 0.15}
    
    final_hybrid_vol = pd.Series(np.nan, index=df.index) # Initialize with NaNs
    
    # Weighted average where models are available
    # temp_weighted_sum = pd.Series(0.0, index=df.index)
    # temp_total_weight = pd.Series(0.0, index=df.index)
    
    # for model_name in weights.keys(): # Iterate in defined order of weights
    #     if model_name in combined_vol_df.columns and not combined_vol_df[model_name].isnull().all():
    #         # Align and add weighted values
    #         current_model_vols = combined_vol_df[model_name].dropna()
    #         temp_weighted_sum = temp_weighted_sum.add(current_model_vols * weights[model_name], fill_value=0)
    #         temp_total_weight.loc[current_model_vols.index] += weights[model_name]
            
    # final_hybrid_vol = temp_weighted_sum / temp_total_weight.replace(0, np.nan)

    # Alternative weighted average: row-wise to handle NaNs per model per day
    for idx in combined_vol_df.index:
        row_values = combined_vol_df.loc[idx].dropna()
        if not row_values.empty:
            current_weights_sum = sum(weights[model_name] for model_name in row_values.index)
            if current_weights_sum > 0:
                weighted_val = sum(row_values[model_name] * weights[model_name] for model_name in row_values.index) / current_weights_sum
                final_hybrid_vol.loc[idx] = weighted_val
    
    final_hybrid_vol = final_hybrid_vol.ffill()
    if final_hybrid_vol.isnull().all(): # If all models failed or resulted in NaNs
        return fallback_vol_func(df.copy(), window=fallback_window)
        
    return final_hybrid_vol