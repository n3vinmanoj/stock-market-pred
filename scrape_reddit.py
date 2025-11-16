import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import praw
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

warnings.filterwarnings('ignore')

# --- Constants ---
OUTPUT_FILE = "reddit_article_sentiments.pkl"

# --- ENTER YOUR REDDIT API KEYS HERE ---
CLIENT_ID = ""
CLIENT_SECRET = ""
USER_AGENT = "MyStockScraper v1.0 by /u/" # Change this

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

def setup_reddit():
    """Initializes and returns a PRAW instance."""
    print("Connecting to Reddit API...")
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        reddit.user.me()
        print("Reddit API connection successful.")
        return reddit
    except Exception as e:
        print(f"--- ERROR: Could not connect to Reddit. ---")
        print(f"       Reason: {e}")
        print("Please check your CLIENT_ID, CLIENT_SECRET, and USER_AGENT variables.")
        return None

def scrape_reddit_data(reddit):
    """
    Scrapes historical post titles and bodies from stock-related subreddits.
    """
    # We will search for common keywords
    SEARCH_QUERY = "stocks OR market OR investing OR earnings"
    # We will get data from the last year
    TIME_FILTER = "year"
    # We will get 1000 posts from each subreddit
    POST_LIMIT = 1000
    
    subreddits_to_scrape = ['stocks', 'StockMarket', 'wallstreetbets']
    article_list = []

    print(f"Scraping posts from the last '{TIME_FILTER}' matching '{SEARCH_QUERY}'")
    
    for sub in subreddits_to_scrape:
        print(f"\n--- Scraping r/{sub} ---")
        subreddit = reddit.subreddit(sub)
        
        # Use tqdm for a progress bar
        for submission in tqdm(subreddit.search(SEARCH_QUERY, time_filter=TIME_FILTER, limit=POST_LIMIT), desc=f"Processing r/{sub}"):
            
            # Combine title and body text
            title = submission.title
            body = submission.selftext
            full_text = title + " " + body
            
            if not full_text.strip():
                continue # Skip empty posts

            # Get BERT sentiment
            sentiment = get_bert_sentiment(full_text)
            
            article_data = {
                'ticker': 'GENERAL', 
                'publish_date': pd.to_datetime(submission.created_utc, unit='s'),
                'title': title,
                'body_text': full_text,
                'url': submission.permalink,
                'neg': sentiment['neg'],
                'neu': sentiment['neu'],
                'pos': sentiment['pos'],
                'compound': sentiment['compound']
            }
            article_list.append(article_data)
            
    return pd.DataFrame(article_list)

def main():
    reddit = setup_reddit()
    if reddit is None:
        return

    try:
        article_sentiments_df = scrape_reddit_data(reddit)
        
        print("\nScraping complete.")
        
        if article_sentiments_df.empty:
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