import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from twelvedata import TDClient
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import warnings
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SENTIMENT_FILE = "daily_sentiment_score.pkl"
SEQ_LENGTH = 30

# --- NEW MULTI-STOCK CONFIG ---
# This list MUST match the list in your train.py
STOCKS_TO_PREDICT = ["AAPL", "MSFT", "GOOGL", "AMZN"] # Add your 10-15 stocks here
MODEL_OUTPUT_DIR = "trained_models"

# --- NEW DYNAMIC LOADER ---
@st.cache_resource
def load_artifacts_for_symbol(symbol):
    """
    Loads all models, scalers, and feature lists for a SPECIFIC symbol.
    """
    print(f"Loading artifacts for symbol: {symbol}...")
    try:
        models = {
            "LSTM": load_model(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_lstm_model.keras")),
            "Random Forest": joblib.load(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_rf_model.joblib")),
            "LightGBM": joblib.load(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_lgbm_model.joblib")),
            "SVR": joblib.load(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_svr_model.joblib"))
        }
        
        scalers = {
            "LSTM": joblib.load(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_data_scaler_lstm.joblib")),
            "Tabular": joblib.load(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_data_scaler_tabular.joblib"))
        }
        
        features = {
            "LSTM": pd.read_json(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_feature_list_lstm.json"), typ='series').tolist(),
            "Tabular": pd.read_json(os.path.join(MODEL_OUTPUT_DIR, f"{symbol}_feature_list_tabular.json"), typ='series').tolist()
        }
        
        print(f"Successfully loaded all artifacts for {symbol}")
        return models, scalers, features
        
    except FileNotFoundError as e:
        st.error(f"Error loading model file for {symbol}: {e}")
        st.error(f"Please make sure you have run train.py for {symbol} and the files are in the '{MODEL_OUTPUT_DIR}' folder.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading artifacts for {symbol}: {e}")
        return None, None, None

# --- HELPER FUNCTIONS (Unchanged) ---

@st.cache_data
def fetch_stock_data(api_key, symbol):
    """Fetches and prepares stock data."""
    td = TDClient(apikey=api_key)
    df = td.time_series(symbol=symbol, interval="1day", outputsize=5000).as_pandas()
    if df.empty:
        return pd.DataFrame()
    df = df.iloc[::-1].reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df[['open', 'high', 'low', 'close', 'volume']]

@st.cache_data
def load_sentiment_data(filepath):
    """Loads pre-processed daily sentiment."""
    try:
        df = pd.read_pickle(filepath)
        df.index = df.index.tz_localize(None) # Remove timezone
        return df[['sentiment_compound_mean', 'post_count']]
    except FileNotFoundError:
        st.error(f"Sentiment file not found: {filepath}. App may not function.")
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

def run_processing_pipeline(df_stock, df_sentiment, merge_type):
    """
    Runs the full data processing pipeline based on merge_type.
    """
    if merge_type == 'left':
        df_merged = pd.merge(df_stock, df_sentiment, left_index=True, right_index=True, how='left')
        df_merged['sentiment_compound_mean'] = df_merged['sentiment_compound_mean'].ffill().fillna(0)
        df_merged['post_count'] = df_merged['post_count'].ffill().fillna(0)
        df_merged = df_merged.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    elif merge_type == 'inner':
        df_merged = pd.merge(df_stock, df_sentiment, left_index=True, right_index=True, how='inner')
        df_merged = df_merged.dropna()
    else:
        raise ValueError("merge_type must be 'left' or 'inner'")

    if df_merged.empty:
        st.error(f"No data found after '{merge_type}' merge. Cannot proceed.")
        return None
        
    df_processed = add_technical_indicators(df_merged)
    return df_processed


def create_sequences(data, seq_length, target_col_index):
    """Creates sequences for time-series models."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col_index])
    return np.array(X), np.array(y)

def invert_scale(scaler, scaled_preds, X_test_data, target_col_index, num_features):
    """
    Inverts the scaled predictions back to actual prices.
    """
    dummy_arr = np.zeros((len(scaled_preds), num_features))
    last_timesteps = X_test_data[:, -1, :]
    
    for i in range(num_features):
        if i == target_col_index:
            dummy_arr[:, i] = scaled_preds
        else:
            dummy_arr[:, i] = last_timesteps[:, i]
            
    inverted_arr = scaler.inverse_transform(dummy_arr)
    return inverted_arr[:, target_col_index]

# --- MAIN APP ---

def main():
    st.title("Multi-Stock Price Prediction")

    API_KEY = st.text_input("Enter your Twelve Data API Key", value="YOUR_API_KEY_HERE")
    
    # --- MODIFIED: Use Selectbox ---
    symbol = st.selectbox("Select Stock Symbol", options=STOCKS_TO_PREDICT)
    
    model_choice = st.selectbox("Select Model", options=["LSTM", "Random Forest", "LightGBM", "SVR"])

    if st.button("Run Prediction") and symbol and API_KEY != "YOUR_API_KEY_HERE":
        
        # --- MODIFIED: Load artifacts for the selected symbol ---
        with st.spinner(f"Loading specialist models for {symbol}..."):
            models, scalers, features = load_artifacts_for_symbol(symbol)
        
        if models is None:
            st.error(f"Failed to load models for {symbol}. See error above.")
            st.stop()
        
        with st.spinner(f"Fetching data for {symbol}..."):
            df_stock = fetch_stock_data(API_KEY, symbol)
            df_sentiment = load_sentiment_data(SENTIMENT_FILE)

        if df_stock.empty:
            st.error(f"No data found for symbol '{symbol}'.")
            return

        st.subheader(f"Historical OHLC Prices for {symbol}")
        st.line_chart(df_stock[['open', 'high', 'low', 'close']])

        # --- DYNAMIC PIPELINE ---
        
        if model_choice == 'LSTM':
            merge_type = 'left'
            model = models['LSTM']
            scaler = scalers['LSTM']
            feature_list = features['LSTM']
        else:
            merge_type = 'inner'
            model = models[model_choice]
            scaler = scalers['Tabular']
            feature_list = features['Tabular']

        st.info(f"Running prediction with {model_choice} (using '{merge_type}' data pipeline).")
        
        df_processed = run_processing_pipeline(df_stock, df_sentiment, merge_type)
        
        if df_processed is None:
            return

        try:
            data_to_scale = df_processed[feature_list]
        except KeyError as e:
            st.error(f"Data features mismatch. Missing feature: {e}. Retrain model.")
            return
            
        data_scaled = scaler.transform(data_to_scale)

        target_col_index = feature_list.index('close')
        X_test, y_test_scaled = create_sequences(data_scaled, SEQ_LENGTH, target_col_index)
        
        if len(X_test) == 0:
            st.error(f"Not enough data to create test sequences (need > {SEQ_LENGTH} days).")
            return
        
        predictions_actual = None
        y_test_actual = None
        y_pred_scaled = None

        with st.spinner(f"Running prediction with {model_choice}..."):
            if model_choice == "LSTM":
                y_pred_scaled = model.predict(X_test).flatten()
            else:
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                y_pred_scaled = model.predict(X_test_flat)

        num_features = len(feature_list)
        y_test_actual = invert_scale(scaler, y_test_scaled, X_test, target_col_index, num_features)
        predictions_actual = invert_scale(scaler, y_pred_scaled, X_test, target_col_index, num_features)

        rmse = np.sqrt(np.mean((y_test_actual - predictions_actual) ** 2))
        st.success(f"Model '{model_choice}' prediction complete. Test RMSE: {rmse:.4f}")

        st.subheader("Actual vs Predicted Closing Prices")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test_actual, label='Actual Close', linewidth=2, color='blue')
        ax.plot(predictions_actual, label='Predicted Close', linestyle='--', color='red')
        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel('Price')
        ax.set_title(f"{symbol} Closing Price Prediction - {model_choice}")
        ax.legend()
        st.pyplot(fig)

    elif API_KEY == "YOUR_API_KEY_HERE":
        st.warning("Please enter your Twelve Data API Key to begin.")

if __name__ == "__main__":
    main()