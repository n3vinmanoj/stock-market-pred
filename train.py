import pandas as pd
import numpy as np
import joblib
import os
from twelvedata import TDClient
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
API_KEY = ""  # Set your Twelve Data API key
SENTIMENT_FILE = "daily_sentiment_score.pkl"
SEQ_LENGTH = 30

# --- NEW MULTI-STOCK CONFIG ---
# Define all the stocks you want to train models for
STOCKS_TO_TRAIN = ["AAPL", "MSFT", "GOOGL", "AMZN"] # Add your 10-15 stocks here
MODEL_OUTPUT_DIR = "trained_models"

# --- HELPER FUNCTIONS ---

def fetch_stock_data(api_key, symbol):
    """Fetches and prepares stock data for a given symbol."""
    print(f"Fetching stock data for {symbol}...")
    td = TDClient(apikey=api_key)
    df = td.time_series(symbol=symbol, interval="1day", outputsize=5000).as_pandas()
    if df.empty:
        print(f"--- WARNING: No stock data found for {symbol}. Skipping. ---")
        return pd.DataFrame()
    df = df.iloc[::-1].reset_index()  # Reverse to chronological order
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df[['open', 'high', 'low', 'close', 'volume']]

def load_sentiment_data(filepath):
    """Loads pre-processed daily sentiment."""
    print(f"Loading sentiment data from {filepath}...")
    try:
        df = pd.read_pickle(filepath)
        df.index = df.index.tz_localize(None) # Remove timezone info for clean merge
        return df[['sentiment_compound_mean', 'post_count']]
    except FileNotFoundError:
        print(f"--- ERROR: Sentiment file not found: {filepath} ---")
        return pd.DataFrame()

def add_technical_indicators(df):
    """
    Adds a select few high-value technical indicators.
    """
    print("Adding select technical indicators...")
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['trend_macd'] = macd.macd()
    df['trend_macd_signal'] = macd.macd_signal()
    df['trend_macd_diff'] = macd.macd_diff()
    df['momentum_rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['volatility_bbm'] = bb.bollinger_mavg()
    df['volatility_bbh'] = bb.bollinger_hband()
    df['volatility_bbl'] = bb.bollinger_lband()
    df = df.fillna(0)
    print(f"Technical indicators added. {len(df)} rows remaining.")
    return df

def create_sequences(data, seq_length, target_col_index):
    """Creates sequences for time-series models."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col_index])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, num_features, lstm_units=64, dropout_rate=0.15):
    """Builds the LSTM model structure."""
    model = Sequential([
        LSTM(lstm_units, input_shape=(seq_length, num_features), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# --- MAIN ---

def main():
    # --- Ensure output directory exists ---
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        print(f"Created directory: {MODEL_OUTPUT_DIR}")

    # Load general market sentiment ONCE
    df_sentiment_base = load_sentiment_data(SENTIMENT_FILE)

    # --- Loop over every stock ---
    for symbol in STOCKS_TO_TRAIN:
        print(f"\n\n{'='*60}")
        print(f"--- STARTING TRAINING FOR SYMBOL: {symbol} ---")
        print(f"{'='*60}")
        
        # Load stock-specific data
        df_stock_base = fetch_stock_data(API_KEY, symbol)
        if df_stock_base.empty:
            continue # Skip this stock if no data

        # --- Define symbol-specific file paths ---
        lstm_model_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_lstm_model.keras")
        rf_model_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_rf_model.joblib")
        lgbm_model_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_lgbm_model.joblib")
        svr_model_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_svr_model.joblib")
        
        scaler_lstm_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_data_scaler_lstm.joblib")
        feature_list_lstm_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_feature_list_lstm.json")
        
        scaler_tabular_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_data_scaler_tabular.joblib")
        feature_list_tabular_path = os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_feature_list_tabular.json")

        # ==============================================================
        # --- PIPELINE 1: LSTM (Large Data, 'left' join) ---
        # ==============================================================
        print(f"\n--- {symbol}: Starting PIPELINE 1: LSTM ---")
        
        df_merged_lstm = pd.merge(df_stock_base, df_sentiment_base, left_index=True, right_index=True, how='left')
        df_merged_lstm['sentiment_compound_mean'] = df_merged_lstm['sentiment_compound_mean'].ffill().fillna(0)
        df_merged_lstm['post_count'] = df_merged_lstm['post_count'].ffill().fillna(0)
        df_merged_lstm = df_merged_lstm.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        print(f"LSTM Data merged (left join). Using {len(df_merged_lstm)} rows.")

        df_processed_lstm = add_technical_indicators(df_merged_lstm)
        
        features_lstm = list(df_processed_lstm.columns)
        print(f"LSTM data processed. Using {len(features_lstm)} features.")
        pd.Series(features_lstm).to_json(feature_list_lstm_path)

        scaler_lstm = MinMaxScaler()
        data_scaled_lstm = scaler_lstm.fit_transform(df_processed_lstm)
        joblib.dump(scaler_lstm, scaler_lstm_path)
        
        target_col_index_lstm = features_lstm.index('close') 
        X_lstm, y_lstm = create_sequences(data_scaled_lstm, SEQ_LENGTH, target_col_index_lstm)
        
        if len(X_lstm) > 0:
            split_lstm = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
            y_train_lstm, y_test_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]

            print(f"LSTM data prepared: X_train: {X_train_lstm.shape}, X_test: {X_test_lstm.shape}")

            print(f"\n--- Training {symbol} LSTM ---")
            model_lstm = build_lstm_model(SEQ_LENGTH, len(features_lstm))
            model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0, validation_data=(X_test_lstm, y_test_lstm)) # Set verbose=0
            model_lstm.save(lstm_model_path)
            print(f"LSTM model saved to {lstm_model_path}")
        else:
            print(f"--- ERROR ({symbol}): Not enough data to train LSTM. ---")

        # ======================================================================
        # --- PIPELINE 2: TABULAR (Small Data, 'inner' join) ---
        # ======================================================================
        print(f"\n--- {symbol}: Starting PIPELINE 2: RF, LGBM, SVR ---")

        df_merged_tabular = pd.merge(df_stock_base, df_sentiment_base, left_index=True, right_index=True, how='inner')
        df_merged_tabular = df_merged_tabular.dropna()
        
        if df_merged_tabular.empty or len(df_merged_tabular) < (SEQ_LENGTH + 5): # Need a bit more data for CV
            print(f"--- ERROR ({symbol}): Not enough data for Tabular models (inner join failed or too small). Skipping. ---")
            continue

        print(f"Tabular Data merged (inner join). Using {len(df_merged_tabular)} rows.")

        df_processed_tabular = add_technical_indicators(df_merged_tabular)
        
        features_tabular = list(df_processed_tabular.columns)
        print(f"Tabular data processed. Using {len(features_tabular)} features.")
        pd.Series(features_tabular).to_json(feature_list_tabular_path)

        scaler_tabular = MinMaxScaler()
        data_scaled_tabular = scaler_tabular.fit_transform(df_processed_tabular)
        joblib.dump(scaler_tabular, scaler_tabular_path)
        
        target_col_index_tabular = features_tabular.index('close') 
        X_tabular, y_tabular = create_sequences(data_scaled_tabular, SEQ_LENGTH, target_col_index_tabular)
        
        if len(X_tabular) < 20: # Need enough data for split and CV
             print(f"--- ERROR ({symbol}): Not enough sequential data for Tabular models. Skipping. ---")
             continue

        split_tabular = int(len(X_tabular) * 0.8)
        X_train_tabular, X_test_tabular = X_tabular[:split_tabular], X_tabular[split_tabular:]
        y_train_tabular, y_test_tabular = y_tabular[:split_tabular], y_tabular[split_tabular:]
        
        X_train_flat = X_train_tabular.reshape(X_train_tabular.shape[0], -1)
        X_test_flat = X_test_tabular.reshape(X_test_tabular.shape[0], -1)

        print(f"Tabular data prepared: X_train_flat: {X_train_flat.shape}, X_test_flat: {X_test_flat.shape}")

        print(f"\n--- Training {symbol} Random Forest ---")
        model_rf = RandomForestRegressor(n_estimators=350, max_depth=9, min_samples_leaf=3, random_state=42, n_jobs=-1)
        model_rf.fit(X_train_flat, y_train_tabular)
        joblib.dump(model_rf, rf_model_path)
        print(f"Random Forest model saved to {rf_model_path}")

        print(f"\n--- Training {symbol} LightGBM ---")
        model_lgbm = LGBMRegressor(n_estimators=400, max_depth=10, learning_rate=0.03, min_data_in_leaf=8, random_state=42, n_jobs=-1)
        model_lgbm.fit(X_train_flat, y_train_tabular)
        joblib.dump(model_lgbm, lgbm_model_path)
        print(f"LightGBM model saved to {lgbm_model_path}")

        print(f"\n--- Training {symbol} SVR with GridSearchCV ---")
        svr = SVR(kernel='rbf')
        param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1]}
        grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error', verbose=0)
        grid_search.fit(X_train_flat, y_train_tabular)
        
        best_svr_model = grid_search.best_estimator_
        print(f"GridSearchCV finished. Best SVR parameters: {grid_search.best_params_}")
        joblib.dump(best_svr_model, svr_model_path)
        print(f"Best SVR model saved to {svr_model_path}")

    print("\n\n--- ALL STOCKS TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()