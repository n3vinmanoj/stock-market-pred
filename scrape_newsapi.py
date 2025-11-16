import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import requests
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

warnings.filterwarnings('ignore')

# --- Constants ---
OUTPUT_FILE = "newsapi_sentiments.pkl"
SEARCH_QUERY = "SPY OR stock market OR investing" # What to search for

# --- PASTE YOUR KEY HERE ---
NEWS_API_KEY = ""

# --- BERT Model Setup ---
print("Loading BERT-based sentiment model (RoBERTa)...")
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded and running on {device}.")


def preprocess_text(text):
    """Cleans text for the RoBERTa model."""
    text = re.sub(r'@\S+', '', text) # Remove @user mentions
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'#\S+', '', text) # Remove hashtags
    return text.strip()

def get_bert_sentiment(text):
    """
    Analyzes a single piece of text and returns VADER-like scores.
    """
    try:
        # Preprocess and truncate text
        text = preprocess_text(text)
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = softmax(logits, dim=1).squeeze()
        
        # Map RoBERTa outputs to VADER-like scores
        # cardiffnlp model outputs: 0: negative, 1: neutral, 2: positive
        probabilities = probabilities.cpu().numpy()
        
        scores = {
            'neg': probabilities[0],
            'neu': probabilities[1],
            'pos': probabilities[2],
            'compound': probabilities[2] - probabilities[0] # Compound = Positive - Negative
        }
        return scores
        
    except Exception as e:
        # print(f"Error processing text: {e}")
        # Return neutral if model fails
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

def scrape_newsapi_data():
    """
    Scrapes news articles from NewsAPI.
    """
    print(f"Scraping NewsAPI for query: '{SEARCH_QUERY}'")
    
    url = (
        'https://newsapi.org/v2/everything?'
        f'q={SEARCH_QUERY}&'
        'language=en&'
        'sortBy=popularity&'
        f'apiKey={NEWS_API_KEY}'
    )
    
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(f"--- ERROR: Could not connect to NewsAPI. ---")
        print(f"       Reason: {e}")
        return None

    if data['status'] == 'error':
        print(f"--- ERROR: NewsAPI returned an error. ---")
        print(f"       Reason: {data['message']}")
        print("       Please check your API key and that you are on the free plan.")
        return None
        
    articles = data.get('articles', [])
    
    if not articles:
        print("No articles found. Halting.")
        return pd.DataFrame()

    article_list = []

    print(f"--- Processing {len(articles)} articles ---")
    
    for article in tqdm(articles, desc="Processing articles"):
        title = article.get('title')
        content = article.get('content') or "" # Use content, or empty string
        
        if not title:
            continue
            
        full_text = title + " " + content
        
        # Get BERT sentiment
        sentiment = get_bert_sentiment(full_text)
        
        article_data = {
            'ticker': 'GENERAL', # This data is general
            'publish_date': pd.to_datetime(article.get('publishedAt')),
            'title': title,
            'body_text': full_text,
            'url': article.get('url'),
            'neg': sentiment['neg'],
            'neu': sentiment['neu'],
            'pos': sentiment['pos'],
            'compound': sentiment['compound']
        }
        article_list.append(article_data)
            
    return pd.DataFrame(article_list)

def main():
    try:
        article_sentiments_df = scrape_newsapi_data()
        
        if article_sentiments_df is None or article_sentiments_df.empty:
            print(f"No articles were successfully scraped. {OUTPUT_FILE} will not be created.")
        else:
            article_sentiments_df.to_pickle(OUTPUT_FILE)
            print(f"Data saved to {OUTPUT_FILE}. Found {len(article_sentiments_df)} posts.")
            print("\nSample of saved data:")
            print(article_sentiments_df.head())

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
    finally:
        print("Script finished.")

if __name__ == "__main__":
    main()