import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import argparse
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt


from market.Stock import Stock
from trade import Trader

from ai.market_state_encoder import MarketStateEncoderVAE
from ai.dynamics_predictor import DynamicsPredictorLSTM

# --- HELPER FUNCTIONS ---

def prepare_data_for_vae(df: pd.DataFrame):
    """Adds technical indicators and scales data for the VAE."""
    print("Preparing data for VAE...")
    df.ta.strategy("all")
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    original_columns = df.columns
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=original_columns)
    print(f"Data prepared with {scaled_df.shape[1]} features and scaled successfully.")
    return scaled_df, scaler

def prepare_multi_stock_data_for_vae(tickers: list[str]):
    """
    (Re-architected) Fetches data for multiple stocks, processes and cleans each
    one INDIVIDUALLY, and then combines the clean data into a final scaled dataset.
    This approach is more robust to variations in data history between stocks.
    """
    print(f"--- Starting Multi-Stock Data Preparation (Robust Method) for {len(tickers)} tickers ---")
    
    MyStrategy = ta.Strategy(
        name="Universal Market Features",
        description="A collection of robust technical indicators for general market analysis.",
        ta=[
            {"kind": "rsi", "length": 14},
            {"kind": "macd"},
            {"kind": "stoch"},
            {"kind": "ema", "length": 20},
            {"kind": "ema", "length": 50},
            {"kind": "ema", "length": 200}, # Longest lookback is 200
            {"kind": "adx"},
            {"kind": "bbands", "length": 20},
            {"kind": "atr", "length": 14},
            {"kind": "obv"},
        ]
    )

    # List to hold the individually cleaned and processed DataFrames
    all_processed_df_list = []
    
    # --- Step 1: Process each stock in isolation ---
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        stock = Stock(ticker)
        hist_df = stock.get_all_historical_data_accurate()

        # Pre-emptive check: If we don't have enough data for our longest indicator, skip it.
        # EMA(200) needs at least 200 data points to produce a single valid value.
        if len(hist_df) < 201:
            print(f"  -> Skipping {ticker}: Insufficient history ({len(hist_df)} rows) for EMA(200).")
            continue

        # Apply indicators directly to this stock's DataFrame
        # Using inplace=True to modify the DataFrame directly
        hist_df.ta.strategy(MyStrategy, append=True, inplace=True)
        
        # Clean this specific DataFrame by dropping rows with any NaN values
        # This removes the initial rows where long-term indicators haven't kicked in yet
        initial_rows = len(hist_df)
        hist_df.dropna(inplace=True)
        final_rows = len(hist_df)
        
        print(f"  -> Cleaning complete. Removed {initial_rows - final_rows} initial rows. {final_rows} valid rows remain.")

        # If the DataFrame is not empty after cleaning, add it to our list
        if not hist_df.empty:
            all_processed_df_list.append(hist_df)

    # --- Step 2: Combine the clean DataFrames ---
    if not all_processed_df_list:
        raise ValueError("Processing failed for all tickers. No data remains after cleaning. Try using stocks with longer histories or shorter-term indicators.")

    print(f"\nCombining {len(all_processed_df_list)} clean stock datasets...")
    # We can now safely combine them. ignore_index=True is good practice here.
    combined_df = pd.concat(all_processed_df_list, ignore_index=True)
    print(f"Final combined dataset created with {len(combined_df)} total rows.")

    # --- Step 3: Scale the final, guaranteed-clean dataset ---
    # We no longer need to check for emptiness or dropna here, it's already done.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_df)
    
    print("\nMulti-stock data prepared and scaled successfully.")
    return scaled_data, scaler

def create_sequences(data: np.ndarray, sequence_length: int):
    """Transforms a time-series array into sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def create_trading_labels(df: pd.DataFrame, lookahead: int, upper_pct: float, lower_pct: float):
    """Creates trading labels using the Triple-Barrier Method."""
    print(f"Creating trading labels with lookahead={lookahead}, upper={upper_pct}%, lower={lower_pct}%...")
    price = df['Close']
    labels = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(len(df) - lookahead):
        window = price.iloc[i+1 : i+1+lookahead]
        upper_barrier = price.iloc[i] * (1 + upper_pct)
        lower_barrier = price.iloc[i] * (1 - lower_pct)
        if (window >= upper_barrier).any():
            labels.iloc[i] = 1 # Buy
        elif (window <= lower_barrier).any():
            labels.iloc[i] = -1 # Sell
    print("Label distribution:\n", labels.value_counts())
    return labels.to_numpy()

def assemble_feature_matrix(z_vectors, lstm_predictor, sequence_length):
    """Assembles the final feature matrix for the XGBoost model."""
    print("Assembling final feature matrix...")
    X_lstm, _ = create_sequences(z_vectors, sequence_length)
    z_pred_vectors = lstm_predictor.predict(X_lstm)
    z_current_vectors = X_lstm[:, -1, :]
    z_actual_for_surprise = z_vectors[sequence_length:-1]
    z_pred_for_surprise = z_pred_vectors[:-1]
    surprise_vectors = z_actual_for_surprise - z_pred_for_surprise
    z_current_aligned = z_current_vectors[1:]
    z_pred_aligned = z_pred_vectors[1:]
    X_final = np.concatenate([z_current_aligned, z_pred_aligned, surprise_vectors], axis=1)
    print(f"Final feature matrix shape: {X_final.shape}")
    return X_final

# --- TRAINING PIPELINE FUNCTIONS ---

def run_vae_training(config):
    """Handles the VAE training stage using multi-stock data."""
    print("\n" + "="*20 + " STAGE 1: VAE TRAINING (Multi-Stock) " + "="*20)
    
    # Use our new multi-stock data preparation function
    prepared_data, data_scaler = prepare_multi_stock_data_for_vae(config['TICKERS'])
    
    # Save the globally-trained scaler
    os.makedirs(os.path.dirname(config['SCALER_PATH']), exist_ok=True)
    joblib.dump(data_scaler, config['SCALER_PATH'])
    print(f"Global data scaler saved to {config['SCALER_PATH']}")

    # The rest of the training is the same!
    input_dim = prepared_data.shape[1]
    vae = MarketStateEncoderVAE(input_dim=input_dim, latent_dim=config['LATENT_DIM'])
    vae.compile(optimizer='adam')
    vae.fit(prepared_data, epochs=config['VAE_EPOCHS'], batch_size=config['BATCH_SIZE'])
    vae.save(config['VAE_MODEL_PATH'])

def run_lstm_training(config):
    """
    Handles the LSTM training stage using the globally trained VAE
    and data from all specified tickers.
    """
    print("\n" + "="*20 + " STAGE 2: LSTM TRAINING (Multi-Stock) " + "="*20)
    
    # --- 1. Load the pre-trained global models ---
    print("Loading pre-trained VAE encoder and data scaler...")
    encoder, _ = MarketStateEncoderVAE.load(config['VAE_MODEL_PATH'])
    scaler = joblib.load(config['SCALER_PATH'])
    
    if not scaler or not encoder:
        print("Missing global scaler or VAE model. Please run VAE training first. Exiting.")
        return

    # --- 2. Re-use the robust data prep logic to get the raw data ---
    # This ensures the data fed to the VAE for encoding is IDENTICAL to how it was trained.
    # We call our robust function but only need the raw, combined DataFrame it produces.
    
    # This is a simplified version of our robust data prep function, as we only need the final DataFrame
    # before scaling, since we will use the loaded scaler.
    print(f"Fetching and preparing data for all tickers: {config['TICKERS']}")
    MyStrategy = ta.Strategy(
        name="Universal Market Features",
        ta=[
            {"kind": "rsi", "length": 14}, {"kind": "macd"}, {"kind": "stoch"},
            {"kind": "ema", "length": 20}, {"kind": "ema", "length": 50}, {"kind": "ema", "length": 200},
            {"kind": "adx"}, {"kind": "bbands", "length": 20}, {"kind": "atr", "length": 14},
            {"kind": "obv"},
        ]
    )
    all_processed_df_list = []
    for ticker in config['TICKERS']:
        stock = Stock(ticker)
        hist_df = stock.get_all_historical_data_accurate()
        if len(hist_df) < 201:
            continue
        hist_df.ta.strategy(MyStrategy, append=True, inplace=True)
        hist_df.dropna(inplace=True)
        if not hist_df.empty:
            all_processed_df_list.append(hist_df)

    if not all_processed_df_list:
        print("No data could be fetched for LSTM training. Exiting.")
        return
        
    combined_df = pd.concat(all_processed_df_list, ignore_index=True)
    print(f"Data prepared. Total rows for encoding: {len(combined_df)}")

    # --- 3. Use the loaded scaler to transform the data ---
    scaled_data = scaler.transform(combined_df)
    
    # --- 4. Encode the entire dataset into latent vectors ---
    print("Encoding full dataset into latent space...")
    _, _, z_vectors = encoder.predict(scaled_data)
    print(f"Successfully encoded dataset into {z_vectors.shape[0]} z-vectors.")

    # --- 5. Create sequences and train the LSTM ---
    print("Creating sequences for LSTM training...")
    X_train, y_train = create_sequences(z_vectors, config['SEQUENCE_LENGTH'])
    print(f"Created {len(X_train)} sequences.")
    
    lstm_predictor = DynamicsPredictorLSTM(
        latent_dim=config['LATENT_DIM'], 
        sequence_length=config['SEQUENCE_LENGTH']
    )
    lstm_predictor.compile(optimizer='adam', loss='mse')
    
    print("Starting LSTM training...")
    lstm_predictor.fit(X_train, y_train, epochs=config['LSTM_EPOCHS'], batch_size=config['BATCH_SIZE'])
    
    # Save the globally trained LSTM
    lstm_predictor.save(config['LSTM_MODEL_PATH'])

def run_trader_training(config):
    """Handles the final XGBoost trader training and evaluation."""
    print("\n" + "="*20 + " STAGE 3: TRADER TRAINING " + "="*20)
    stock = Stock(config['TICKER'])
    historical_data = stock.get_historical_data_accurate()
    encoder, _ = MarketStateEncoderVAE.load(config['VAE_MODEL_PATH'])
    lstm_predictor = DynamicsPredictorLSTM.load(config['LSTM_MODEL_PATH'])
    scaler = joblib.load(config['SCALER_PATH'])
    if historical_data.empty or not encoder or not lstm_predictor or not scaler:
        print("Missing data or models. Please run previous training stages. Exiting.")
        return

    scaled_df = pd.DataFrame(scaler.transform(historical_data), index=historical_data.index, columns=historical_data.columns)
    _, _, z_vectors = encoder.predict(scaled_df.to_numpy())
    X_final = assemble_feature_matrix(z_vectors, lstm_predictor, config['SEQUENCE_LENGTH'])
    
    label_start_index = config['SEQUENCE_LENGTH'] + 1
    labels_df = historical_data.iloc[label_start_index:]
    y_final = create_trading_labels(labels_df, config['LOOKAHEAD_PERIOD'], config['UPPER_BARRIER_PCT'], config['LOWER_BARRIER_PCT'])
    
    min_len = min(len(X_final), len(y_final))
    X_final, y_final = X_final[:min_len], y_final[:min_len]
    y_mapped = np.where(y_final == -1, 0, np.where(y_final == 0, 1, 2))
    print(f"\nFinal aligned shapes -> X: {X_final.shape}, y: {y_mapped.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_mapped, test_size=0.2, shuffle=False)
    
    xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False, eval_metric='mlogloss')
    xgb_classifier.fit(X_train, y_train)
    xgb_classifier.save_model(config['XGB_MODEL_PATH'])
    print(f"Final XGBoost trading model saved to {config['XGB_MODEL_PATH']}")

    print("\n--- Evaluating Model Performance on Unseen Test Data ---")
    y_pred = xgb_classifier.predict(X_test)
    target_names = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

def evaluate_vae_reconstruction(config):
    """Loads a trained VAE and visually compares original vs. reconstructed data."""
    print("\n" + "="*20 + " VAE EVALUATION " + "="*20)
    # Load the necessary components
    encoder, decoder = MarketStateEncoderVAE.load(config['VAE_MODEL_PATH'])
    scaler = joblib.load(config['SCALER_PATH'])
    stock = Stock(config['TICKER'])
    historical_data = stock.get_historical_data_accurate()

    if not encoder or not decoder or not scaler or historical_data.empty:
        print("Missing components for VAE evaluation. Exiting.")
        return

    # Prepare a sample of the data (e.g., 200 points)
    sample_original_df = historical_data.head(200)
    sample_scaled_data = scaler.transform(sample_original_df)

    # Run the data through the VAE
    _, _, z_vectors = encoder.predict(sample_scaled_data)
    reconstructed_scaled_data = decoder.predict(z_vectors)

    # IMPORTANT: Inverse transform to get back to the original scale for comparison
    reconstructed_df = pd.DataFrame(
        scaler.inverse_transform(reconstructed_scaled_data),
        index=sample_original_df.index,
        columns=sample_original_df.columns
    )

    # Plot the comparison for 'Close' price
    plt.figure(figsize=(15, 7))
    plt.title(f"{config['TICKER']} VAE Reconstruction Quality")
    plt.plot(sample_original_df.index, sample_original_df['Close'], label='Original Close Price', color='blue', alpha=0.7)
    plt.plot(reconstructed_df.index, reconstructed_df['Close'], label='Reconstructed Close Price', color='red', linestyle='--')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    print("VAE Evaluation plot displayed. Close the plot window to continue.")

def evaluate_lstm_prediction(config):
    """Loads a trained LSTM and visually compares its predictions to actuals."""
    print("\n" + "="*20 + " LSTM EVALUATION " + "="*20)
    # Load components
    encoder, _ = MarketStateEncoderVAE.load(config['VAE_MODEL_PATH'])
    lstm_predictor = DynamicsPredictorLSTM.load(config['LSTM_MODEL_PATH'])
    scaler = joblib.load(config['SCALER_PATH'])
    stock = Stock(config['TICKER'])
    historical_data = stock.get_historical_data_accurate()

    if not encoder or not lstm_predictor or not scaler or historical_data.empty:
        print("Missing components for LSTM evaluation. Exiting.")
        return

    # Generate z_vectors
    scaled_data = scaler.transform(historical_data)
    _, _, z_vectors = encoder.predict(scaled_data)

    # Create sequences
    X_seq, y_actual = create_sequences(z_vectors, config['SEQUENCE_LENGTH'])
    
    # Get predictions
    y_pred = lstm_predictor.predict(X_seq)

    # Plot the comparison for the first two dimensions of the z-vector
    plt.figure(figsize=(15, 7))
    plt.title(f"LSTM Prediction vs. Actual (Latent Space Dimension 0)")
    # Plot a slice of the data to keep it readable
    plot_range = range(200)
    plt.plot(plot_range, y_actual[:200, 0], label='Actual z[0]', color='blue', alpha=0.7)
    plt.plot(plot_range, y_pred[:200, 0], label='Predicted z[0]', color='red', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("LSTM Evaluation plot displayed. Close the plot window to continue.")

def run_backtest_simulation(config):
    """
    Runs a full trading simulation on the test data using the trained XGBoost model
    and the Trader environment to calculate financial performance.
    """
    print("\n" + "="*20 + " STAGE 4: BACKTEST SIMULATION " + "="*20)
    # --- 1. Load all necessary components ---
    xgb_trader = xgb.XGBClassifier()
    xgb_trader.load_model(config['XGB_MODEL_PATH'])
    
    encoder, _ = MarketStateEncoderVAE.load(config['VAE_MODEL_PATH'])
    lstm_predictor = DynamicsPredictorLSTM.load(config['LSTM_MODEL_PATH'])
    scaler = joblib.load(config['SCALER_PATH'])
    
    stock = Stock(config['TICKER'])
    historical_data = stock.get_historical_data_accurate()

    if not all([xgb_trader, encoder, lstm_predictor, scaler]) or historical_data.empty:
        print("Missing components for backtest. Please run the full pipeline first. Exiting.")
        return

    # --- 2. Prepare the full feature set and split into train/test ---
    # We need to do this to find the exact start of our test data
    scaled_df, _ = prepare_data_for_vae(historical_data.copy())
    _, _, z_vectors = encoder.predict(scaled_df.to_numpy())
    X_final = assemble_feature_matrix(z_vectors, lstm_predictor, config['SEQUENCE_LENGTH'])
    
    # We need the original historical data for the trader, but aligned with our features
    label_start_index = config['SEQUENCE_LENGTH'] + 1
    min_len = len(X_final)
    aligned_historical_data = historical_data.iloc[label_start_index : label_start_index + min_len]

    # Split the data exactly as it was during training to get the test set
    train_size = int(len(X_final) * 0.8)
    X_test = X_final[train_size:]
    test_data_df = aligned_historical_data.iloc[train_size:]

    # --- 3. Initialize your Trader simulation ---
    trader = Trader(historical_data=test_data_df, initial_balance=10000.0)

    # --- 4. Run the simulation loop ---
    print(f"\nRunning backtest on {len(X_test)} unseen data points...")
    for i in range(len(X_test)):
        # Get the features for the current step
        current_features = X_test[i].reshape(1, -1)
        
        # Use the AI to predict an action {0: Sell, 1: Hold, 2: Buy}
        predicted_action = xgb_trader.predict(current_features)[0]
        
        # Execute the action in the simulation environment
        if predicted_action == 2: # Buy
            # Strategy: Use 10% of available cash for each buy signal
            amount_to_buy = trader.current_balance * 0.10
            trader.buy(amount_to_buy)
        elif predicted_action == 0: # Sell
            # Strategy: Sell all shares on a sell signal
            amount_to_sell = trader.shares_held
            trader.sell(amount_to_sell)
        else: # Hold
            trader.hold()
            
        # Advance the simulation to the next day/minute
        trader.next_step()

    # --- 5. Print the final financial summary ---
    trader.print_summary()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train and manage the multi-stage trading AI.")
        parser.add_argument('--mode', type=str, default='full-pipeline',
                            choices=['train-vae', 'train-lstm', 'train-trader', 'full-pipeline', 'eval-vae', 'eval-lstm', 'backtest'],
                            help='The training mode to run.')
        args = parser.parse_args()

        # --- Global Configuration ---
        config = {
            "TICKERS": ["TSLA", "AAPL", "GOOGL", "MSFT", "NVDA"],
            # Paths
            "VAE_MODEL_PATH": f"ai/models/vae",
            "SCALER_PATH": f"ai/models/scaler.joblib",
            "LSTM_MODEL_PATH": f"ai/models/lstm_predictor.keras",
            "XGB_MODEL_PATH": f"ai/models/xgb_trader.json",
            # Hyperparameters
            "LATENT_DIM": 16,
            "SEQUENCE_LENGTH": 30,
            "VAE_EPOCHS": 50,
            "LSTM_EPOCHS": 50,
            "BATCH_SIZE": 64,
            # Labeling Config
            "LOOKAHEAD_PERIOD": 15,
            "UPPER_BARRIER_PCT": 0.005,
            "LOWER_BARRIER_PCT": 0.005,
        }

        if args.mode == 'train-vae':
            run_vae_training(config)
        elif args.mode == 'train-lstm':
            run_lstm_training(config)
        elif args.mode == 'train-trader':
            run_trader_training(config)
        elif args.mode == 'full-pipeline':
            run_vae_training(config)
            run_lstm_training(config)
            run_trader_training(config)
        elif args.mode == 'eval-vae':
            evaluate_vae_reconstruction(config)
        elif args.mode == 'eval-lstm':
            evaluate_lstm_prediction(config)
        elif args.mode == 'backtest':
            run_backtest_simulation(config)

    except KeyboardInterrupt:
        quit()