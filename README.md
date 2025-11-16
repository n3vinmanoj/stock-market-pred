# Stock Market Prediction with Sentiment Analysis

A comprehensive stock market prediction system that combines technical analysis, sentiment analysis from news and social media, and multiple machine learning models to forecast stock prices.

## Features

- **Multi-Model Approach**: Utilizes LSTM, Random Forest, LightGBM, and SVR models for robust predictions
- **Sentiment Analysis**: Integrates news and social media sentiment data
- **Technical Indicators**: Includes MACD, RSI, and Bollinger Bands
- **Streamlit Web Interface**: User-friendly interface for stock prediction visualization
- **Multi-Stock Support**: Train and predict on multiple stocks (AAPL, MSFT, GOOGL, AMZN by default)

## Project Structure

- [app.py](cci:7://file:///Users/adi/Desktop/final/app.py:0:0-0:0): Main Streamlit application for predictions and visualization
- [train.py](cci:7://file:///Users/adi/Desktop/final/train.py:0:0-0:0): Script for training and saving prediction models
- [prepare_sentiment_data.py](cci:7://file:///Users/adi/Desktop/final/prepare_sentiment_data.py:0:0-0:0): Processes sentiment data from news and social media
- [scrape_newsapi.py](cci:7://file:///Users/adi/Desktop/final/scrape_newsapi.py:0:0-0:0): Fetches and processes news data
- [scrape_reddit.py](cci:7://file:///Users/adi/Desktop/final/scrape_reddit.py:0:0-0:0): Collects and processes Reddit discussions
- [trained_models/](cci:7://file:///Users/adi/Desktop/final/trained_models:0:0-0:0): Directory for storing trained models and scalers

## Prerequisites

- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - lightgbm
  - ta (Technical Analysis library)
  - twelvedata (for stock data)

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Adithyab3103/Project_Fall_2025.git](https://github.com/Adithyab3103/Project_Fall_2025.git)
   cd Project_Fall_2025
