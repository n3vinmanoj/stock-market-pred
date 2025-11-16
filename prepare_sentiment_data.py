import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

REDDIT_FILE = "reddit_article_sentiments.pkl"
NEWSAPI_FILE = "newsapi_sentiments.pkl"
OUTPUT_FILE = "daily_sentiment_score.pkl"

def load_data(filepath):
    """Loads a pickle file safely."""
    try:
        df = pd.read_pickle(filepath)
        print(f"Successfully loaded {filepath}. Found {len(df)} records.")
        return df
    except FileNotFoundError:
        print(f"--- WARNING: File not found: {filepath}. Skipping. ---")
        return pd.DataFrame()
    except Exception as e:
        print(f"--- ERROR loading {filepath}: {e} ---")
        return pd.DataFrame()

def combine_and_process_sentiment():
    """
    Loads Reddit and NewsAPI data, combines them, and aggregates
    by day to create a single daily average sentiment score.
    """
    print("--- Starting Sentiment Data Consolidation ---")
    
    # 1. Load both datasets
    df_reddit = load_data(REDDIT_FILE)
    df_news = load_data(NEWSAPI_FILE)
    
    if df_reddit.empty and df_news.empty:
        print("--- ERROR: No data loaded. Both files might be missing. ---")
        print(f"Please run scrape_reddit.py and scrape_newsapi.py first.")
        return

    # 2. Combine into one DataFrame
    df_combined = pd.concat([df_reddit, df_news], ignore_index=True)
    
    if 'publish_date' not in df_combined.columns:
        print("--- ERROR: 'publish_date' column missing. ---")
        return

    # 3. Ensure 'publish_date' is a datetime object and sort
    #    Use utc=True to correctly handle mixed naive/aware datetimes
    #    This is the FIX:
    df_combined['publish_date'] = pd.to_datetime(df_combined['publish_date'], utc=True)
    
    df_combined = df_combined.sort_values(by='publish_date')
    
    # 4. Set date as index for easy resampling
    # We normalize to 'date' to merge with daily stock data
    # We also make sure the index is timezone-aware (UTC)
    df_combined['date'] = df_combined['publish_date'].dt.tz_convert('UTC').dt.normalize()
    df_combined = df_combined.set_index('date')
    
    if 'compound' not in df_combined.columns:
        print("--- ERROR: 'compound' sentiment score not found. ---")
        return

    # 5. Aggregate by day. We'll take the mean 'compound' score.
    # We also count posts to see how 'busy' a day was.
    print("Aggregating sentiment scores by day...")
    df_daily_sentiment = df_combined.resample('D').agg(
        sentiment_compound_mean=('compound', 'mean'),
        sentiment_compound_sum=('compound', 'sum'),
        post_count=('compound', 'count')
    )
    
    # 6. Fill missing days
    # If no news on a weekend, the sentiment is neutral (0)
    df_daily_sentiment = df_daily_sentiment.fillna(0)
    
    # 7. Save the processed file
    df_daily_sentiment.to_pickle(OUTPUT_FILE)
    
    print("\n--- Sentiment Data Consolidation Complete ---")
    print(f"Saved aggregated daily sentiment to {OUTPUT_FILE}")
    print("\nSample of aggregated data:")
    print(df_daily_sentiment.tail())

if __name__ == "__main__":
    combine_and_process_sentiment()