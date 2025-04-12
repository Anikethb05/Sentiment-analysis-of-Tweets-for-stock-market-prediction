import time
from functools import lru_cache
import yfinance as yf
import pandas_ta as ta
from flask import Flask, jsonify, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import subprocess
import re
from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import random

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Supported stocks
STOCKS = ['TSLA', 'MSFT', 'PG', 'META', 'AMZN', 'GOOG', 'AMD', 'AAPL',
          'NFLX', 'TSM', 'KO', 'F', 'COST', 'DIS', 'VZ', 'CRM', 'INTC', 'BA',
          'BX', 'NOC', 'PYPL', 'ENPH', 'NIO', 'ZS', 'XPEV']

FEATURE_TEMPLATE = ['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
                    'EMA', 'logmomentum', 'sentiment_score', 'Negative', 'Neutral', 'Positive']

MODEL_DIR = 'models'

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def get_sentiment(text):
    sentiment = analyzer.polarity_scores(str(text))
    return pd.Series([sentiment['compound'], sentiment['neg'], sentiment['neu'], sentiment['pos']])

def get_tech_ind(d):
    data = d.copy()
    data['MA7'] = data.iloc[:,4].rolling(window=7).mean()
    data['MA20'] = data.iloc[:,4].rolling(window=20).mean() 
    data['MA10'] = data['Close'].rolling(window=10).mean()

    data['MACD'] = data.iloc[:,4].ewm(span=26).mean() - data.iloc[:,1].ewm(span=12,adjust=False).mean()
    #This is the difference of Closing price and Opening Price

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 4].rolling(20).std()
    data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA20'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:,4].ewm(com=0.5).mean()

    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:,4] - 1)

    return data

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

def train_model(stock):
    stock_name = stock.upper()
    model_path = os.path.join(MODEL_DIR, f"{stock_name}_model.h5")
    scaler_x_path = os.path.join(MODEL_DIR, f"{stock_name}_scaler_x.pkl")
    scaler_y_path = os.path.join(MODEL_DIR, f"{stock_name}_scaler_y.pkl")
    last_trained_path = os.path.join(MODEL_DIR, f"{stock_name}_last_trained.pkl")

    # Check if model was trained today
    current_date = datetime.now().date()
    retrain = True
    if os.path.exists(last_trained_path):
        try:
            with open(last_trained_path, 'rb') as f:
                last_trained = pickle.load(f)
            if last_trained.date() == current_date:
                retrain = False
                print(f"Model for {stock_name} was trained today ({last_trained.date()}), loading existing model")
        except Exception as e:
            print(f"Error reading last trained timestamp for {stock_name}: {e}")

    # Load existing model if no retraining is needed
    if not retrain and os.path.exists(model_path):
        try:
            model = load_model(model_path)
            with open(scaler_x_path, 'rb') as f:
                scaler_x = pickle.load(f)
            with open(scaler_y_path, 'rb') as f:
                scaler_y = pickle.load(f)
            print(f"Loaded existing model and scalers for {stock_name}")
            return model, scaler_x, scaler_y
        except Exception as e:
            print(f"Error loading model for {stock_name}: {e}")
            retrain = True

    # Step 1: Load existing data
    df = pd.read_csv('stock_tweets.csv')
    df[['sentiment_score', 'Negative', 'Neutral', 'Positive']] = df['Tweet'].apply(get_sentiment)
    
    # Step 2: Fetch additional tweets using snscrape
    print(f"Fetching additional tweets for {stock_name} using snscrape...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"${stock_name} OR #{stock_name} OR {stock_name} stock lang:en since:{start_str} until:{end_str}"
        command = ["snscrape", "--jsonl", "--max-results", "100", "twitter-search", query]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()
        
        new_tweets = []
        for line in output.strip().split('\n'):
            if line:
                tweet_data = json.loads(line)
                new_tweets.append({
                    'Date': tweet_data['date'].split('T')[0],
                    'Tweet': tweet_data['content'],
                    'Stock Name': stock_name
                })
        
        if new_tweets:
            new_df = pd.DataFrame(new_tweets)
            new_df[['sentiment_score', 'Negative', 'Neutral', 'Positive']] = new_df['Tweet'].apply(get_sentiment)
            df = pd.concat([df, new_df], ignore_index=True)
            print(f"Added {len(new_tweets)} new tweets from snscrape")
        else:
            print("No new tweets found from snscrape")
            
    except Exception as e:
        print(f"Error fetching tweets from snscrape: {e}")
    
    # Step 3: Filter data for the specific stock
    stock_tweets_df = df[df['Stock Name'] == stock_name].copy()
    stock_tweets_df['Date'] = pd.to_datetime(stock_tweets_df['Date'])
    stock_tweets_df['Date'] = stock_tweets_df['Date'].dt.date
    
    daily_sentiment = stock_tweets_df.groupby('Date').mean(numeric_only=True)
    
    # Step 4: Load stock price data
    all_stocks = pd.read_csv('stock_yfinance_data.csv')
    stock_df = all_stocks[all_stocks['Stock Name'] == stock_name]
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # Step 5: Get fresh technical indicators from Yahoo Finance
    print(f"Fetching latest technical indicators for {stock_name}...")
    try:
        yf_data = yf.download(stock_name, period='90d', interval='1d')
        
        yf_data['MA7'] = yf_data['Close'].rolling(window=7).mean()
        yf_data['MA10'] = yf_data['Close'].rolling(window=10).mean()
        yf_data['MA20'] = yf_data['Close'].rolling(window=20).mean()
        
        exp1 = yf_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = yf_data['Close'].ewm(span=26, adjust=False).mean()
        yf_data['MACD'] = exp1 - exp2
        
        yf_data['20SD'] = yf_data['Close'].rolling(window=20).std()
        middle_band = yf_data['Close'].rolling(window=20).mean()
        yf_data['upper_band'] = middle_band + (yf_data['20SD'] * 2)
        yf_data['lower_band'] = middle_band - (yf_data['20SD'] * 2)
        
        yf_data['EMA'] = yf_data['Close'].ewm(span=10, adjust=False).mean()
        
        yf_data = yf_data.reset_index()
        yf_data['Date'] = pd.to_datetime(yf_data['Date']).dt.date
        
        stock_df = yf_data
        stock_df['Stock Name'] = stock_name
        
    except Exception as e:
        print(f"Error fetching fresh technical indicators: {e}")
        tech_df = get_tech_ind(stock_df)
        stock_df = tech_df.iloc[20:,:].reset_index(drop=True)
    
    # Step 6: Merge stock data with sentiment data
    stock_df.set_index('Date', inplace=True)
    merged_df = stock_df.merge(daily_sentiment, left_index=True, right_index=True, how='inner')
    
    # Step 7: Prepare data for modeling
    scaled_df = merged_df[['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
       'EMA', 'logmomentum','sentiment_score', 'Negative', 'Neutral', 'Positive','Close']]
    
    scaler = MinMaxScaler()
    scaled_df[['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
       'EMA', 'logmomentum','sentiment_score', 'Negative', 'Neutral', 'Positive','Close']] = scaler.fit_transform(scaled_df[['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
       'EMA', 'logmomentum','sentiment_score', 'Negative', 'Neutral', 'Positive','Close']])
    
    features = scaled_df[['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
       'EMA', 'logmomentum','sentiment_score', 'Negative', 'Neutral', 'Positive']]
    target = scaled_df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    features_list = ['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
       'EMA', 'logmomentum','sentiment_score', 'Negative', 'Neutral', 'Positive']
    target = 'Close'

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(merged_df[features_list])

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(merged_df[[target]])

    scaled_data = np.hstack((X_scaled, y_scaled))

    seq_length = 10
    X, y = create_sequences(scaled_data, seq_length)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = 1

    # Step 8: Build and train model
    model = tf.keras.Sequential([
        LSTM(units=1024, return_sequences=True, input_shape=(input_dim, feature_size), recurrent_dropout=0.3),
        LSTM(units=512, return_sequences=True, recurrent_dropout=0.3),
        LSTM(units=256, return_sequences=True, recurrent_dropout=0.3),
        LSTM(units=128, return_sequences=True, recurrent_dropout=0.3),
        LSTM(units=64, recurrent_dropout=0.3),
        Dense(32),
        Dense(16),
        Dense(8),
        Dense(units=output_dim)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
    
    print(f"Model training complete for {stock_name}")
    
    try:
        model.save(model_path)
        with open(scaler_x_path, 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(scaler_y_path, 'wb') as f:
            pickle.dump(scaler_y, f)
        with open(last_trained_path, 'wb') as f:
            pickle.dump(datetime.now(), f)
        print(f"Saved model, scalers, and timestamp for {stock_name}")
    except Exception as e:
        print(f"Error saving model or timestamp for {stock_name}: {e}")

    return model, scaler_X, scaler_y

def fetch_news_data(ticker):
    """Fetch stock news from Yahoo Finance with sentiment analysis."""
    try:
        print(f"Fetching Yahoo Finance news for {ticker}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        articles = soup.find_all("div", {"class": "Ov(h) Pend(44px) Pstart(25px)"})
        if not articles:
            articles = soup.find_all("h3", {"class": "Mb(5px)"})
        
        for article in articles[:15]:
            if hasattr(article, 'text') and article.text:
                text = article.text.strip()
                sentiment = analyzer.polarity_scores(text)
                news_items.append({
                    'text': text,
                    'link': '',
                    'sentiment': {
                        'compound': round(sentiment['compound'], 4),
                        'negative': round(sentiment['neg'], 4),
                        'neutral': round(sentiment['neu'], 4),
                        'positive': round(sentiment['pos'], 4)
                    }
                })
        
        if not news_items:
            all_text = soup.get_text()
            relevant_sentences = [s.strip() for s in all_text.split('.') if ticker in s and len(s) > 30]
            for text in relevant_sentences[:15]:
                sentiment = analyzer.polarity_scores(text)
                news_items.append({
                    'text': text,
                    'link': '',
                    'sentiment': {
                        'compound': round(sentiment['compound'], 4),
                        'negative': round(sentiment['neg'], 4),
                        'neutral': round(sentiment['neu'], 4),
                        'positive': round(sentiment['pos'], 4)
                    }
                })
        
        print(f"Found {len(news_items)} news items for {ticker}")
        if news_items:
            for i, item in enumerate(news_items[:3]):
                print(f"- News {i+1}: {item['text'][:100]}...")
                
        return news_items if news_items else []
    except Exception as e:
        print(f"Error fetching Yahoo Finance news: {e}")
        return []

def fetch_alternative_sentiment(ticker):
    """A backup approach to get sentiment data for a stock using financial websites."""
    sources = [
        f"https://finance.yahoo.com/quote/{ticker}",
        f"https://www.marketwatch.com/investing/stock/{ticker}",
        f"https://seekingalpha.com/symbol/{ticker}"
    ]
    
    all_texts = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Try each source
        for url in sources:
            try:
                response = requests.get(url, headers=headers, timeout=8)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract title and meta description
                    title = soup.find('title')
                    if title and title.text:
                        all_texts.append(title.text)
                    
                    # Get meta description
                    meta_desc = soup.find('meta', {'name': 'description'})
                    if meta_desc and meta_desc.get('content'):
                        all_texts.append(meta_desc.get('content'))
                    
                    # Extract paragraphs that mention the ticker
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        if p.text and ticker in p.text and len(p.text) > 20:
                            all_texts.append(p.text.strip())
                    
            except Exception as e:
                print(f"Error fetching from {url}: {e}")
                continue
        
        print(f"Fetched {len(all_texts)} text snippets from financial websites for {ticker}")
        return all_texts
        
    except Exception as e:
        print(f"Error in alternative sentiment fetch: {e}")
        return []

@lru_cache(maxsize=32)
def fetch_sentiment_data(stock_name):
    """Fetch sentiment data using multiple methods."""
    # First try news data
    news_items = fetch_news_data(stock_name)
    
    # If we got some news, use that
    if len(news_items) >= 5:
        return news_items
    
    # Otherwise try alternative sources
    alternative_data = fetch_alternative_sentiment(stock_name)
    
    # Combine whatever we got
    combined_data = news_items + alternative_data
    
    # If we still don't have enough data
    if len(combined_data) < 5:
        print(f"Warning: Limited sentiment data for {stock_name} ({len(combined_data)} items)")
        
    return combined_data

def get_sentiment_scores(texts):
    """Calculate sentiment scores with improved error handling."""
    if not texts:
        print("Warning: No texts available for sentiment analysis")
        # Return neutral sentiment with slight negative bias
        return 0,0,0,0
    
    try:
        compound_scores = []
        positives = []
        neutrals = []
        negatives = []
        
        for text in texts:
            if not text or not isinstance(text, str):
                continue
                
            scores = analyzer.polarity_scores(text)
            compound_scores.append(scores['compound'])
            positives.append(scores['pos'])
            neutrals.append(scores['neu'])
            negatives.append(scores['neg'])
            
        # Calculate averages, handling empty lists
        if not compound_scores:
            print("Warning: No valid sentiment scores calculated")
            return 0,0,0,0
            
        sentiment_score = np.mean(compound_scores)
        positive = np.mean(positives)
        neutral = np.mean(neutrals)
        negative = np.mean(negatives)
        
        print(f"Sentiment analysis complete - compound: {sentiment_score:.4f}, neg: {negative:.4f}, neu: {neutral:.4f}, pos: {positive:.4f}")
        return sentiment_score, negative, neutral, positive

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # Return slightly negative sentiment on error (market often defaults to caution)
        return 0,0,0,0

def fetch_tweets(stock_name):
    try:
        stock_name = stock_name.upper()
        print(f"Fetching tweets for {stock_name} for today...")
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        start_str = today.strftime('%Y-%m-%d')
        end_str = tomorrow.strftime('%Y-%m-%d')
        
        query = f"${stock_name} OR #{stock_name} OR {stock_name} stock lang:en since:{start_str} until:{end_str}"
        command = ["snscrape", "--jsonl", "--max-results", "50", "twitter-search", query]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()
        
        tweets = []
        for line in output.strip().split('\n'):
            if line:
                tweet_data = json.loads(line)
                tweet_date = datetime.strptime(tweet_data['date'].split('T')[0], '%Y-%m-%d').date()
                if tweet_date == today:
                    sentiment = analyzer.polarity_scores(tweet_data['content'])
                    tweets.append({
                        'text': tweet_data['content'],
                        'link': tweet_data.get('url', ''),
                        'sentiment': {
                            'compound': round(sentiment['compound'], 4),
                            'negative': round(sentiment['neg'], 4),
                            'neutral': round(sentiment['neu'], 4),
                            'positive': round(sentiment['pos'], 4)
                        }
                    })
        
        print(f"Found {len(tweets)} tweets for {stock_name} today")
        return tweets if tweets else [{'text': 'No tweets found for today', 'link': '', 'sentiment': {'compound': 0, 'negative': 0, 'neutral': 0, 'positive': 0}}]
    except Exception as e:
        print(f"Error fetching tweets for {stock_name}: {e}")
        return [{'text': 'Error fetching tweets', 'link': '', 'sentiment': {'compound': 0, 'negative': 0, 'neutral': 0, 'positive': 0}}]

def get_technical_indicators(symbol):
    try:
        df = yf.download(symbol, period='60d', interval='1d')

        # Flatten multi-index columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 26:
            print(f"Not enough data with 60 days, trying 90 days for {symbol}")
            df = yf.download(symbol, period='90d', interval='1d')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if len(df) < 26:
            raise ValueError(f"Not enough data points for {symbol}, only got {len(df)}")

        df = df.dropna(subset=['Close'])

        features = {}
        features['MA7'] = float(df['Close'].rolling(window=7).mean().iloc[-1])
        features['MA10'] = float(df['Close'].rolling(window=10).mean().iloc[-1])
        features['MA20'] = float(df['Close'].rolling(window=20).mean().iloc[-1])

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        features['MACD'] = float((exp1 - exp2).iloc[-1])

        std_dev = df['Close'].rolling(window=20).std()
        middle_band = df['Close'].rolling(window=20).mean()
        features['20SD'] = float(std_dev.iloc[-1])
        features['upper_band'] = float((middle_band + 2 * std_dev).iloc[-1])
        features['lower_band'] = float((middle_band - 2 * std_dev).iloc[-1])

        features['EMA'] = float(df['Close'].ewm(span=10, adjust=False).mean().iloc[-1])

        current_close = float(df['Close'].iloc[-1])
        past_close = float(df['Close'].iloc[-11]) if len(df) >= 11 else float(df['Close'].iloc[0])
        momentum = current_close - past_close
        features['logmomentum'] = (
            float(np.log1p(momentum)) if momentum > 0 else
            float(-np.log1p(abs(momentum))) if momentum < 0 else 0.0
        )

        features['Close'] = current_close

        # Check for missing values
        if any(pd.isna(v) for v in features.values()):
            missing = [k for k, v in features.items() if pd.isna(v)]
            raise ValueError(f"Missing values in indicators: {missing}")

        print("Technical indicators fetched successfully")
        return features
    except Exception as e:
        print(f"Error in technical indicators: {e}")
        return None

def get_historical_data(symbol, days):
    try:
        period = f"{max(int(days), 30)}d"
        df = yf.download(symbol, period=period, interval='1d')

        # Flatten multi-index columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise ValueError(f"No data fetched for {symbol}")

        if 'Close' not in df.columns:
            raise ValueError("Missing 'Close' column in fetched data")

        # Drop rows with NaN in Close and check minimum length
        df = df.dropna(subset=['Close'])
        if len(df) < 20:
            raise ValueError(f"Not enough valid data for indicators: only {len(df)} valid rows")

        # Calculate technical indicators with forward fill for initial NaN
        df['MA7'] = df['Close'].rolling(window=7, min_periods=1).mean().fillna(method='ffill').fillna(df['Close'].iloc[0])
        df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean().fillna(method='ffill').fillna(df['Close'].iloc[0])
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean().fillna(method='ffill').fillna(df['Close'].iloc[0])

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (exp1 - exp2).fillna(0)

        df['20SD'] = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
        df['middle_band'] = df['Close'].rolling(window=20, min_periods=1).mean().fillna(method='ffill').fillna(df['Close'].iloc[0])
        df['upper_band'] = df['middle_band'] + (df['20SD'] * 2)
        df['lower_band'] = df['middle_band'] - (df['20SD'] * 2)

        df['EMA'] = df['Close'].ewm(span=10, adjust=False).mean().fillna(method='ffill').fillna(df['Close'].iloc[0])

        # Prepare data for return
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        required_cols = ['Close', 'MA7', 'MA10', 'MA20', 'MACD', 'upper_band', 'lower_band', 'EMA']
        data = df[['Date'] + required_cols].to_dict(orient='records')

        print(f"Processed {len(data)} historical records for {symbol}")
        return data

    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return [{
            'Date': 'N/A', 'Close': 0, 'MA7': 0, 'MA10': 0, 'MA20': 0,
            'MACD': 0, 'upper_band': 0, 'lower_band': 0, 'EMA': 0
        }]
    
@app.route('/predict/')
def index():
    return render_template('index.html', stocks=STOCKS)

@app.route('/predict/<stock_name>')
def predict(stock_name):
    start_time = time.time()
    stock_name = stock_name.upper()
    if stock_name not in STOCKS:
        return render_template('predict.html', error="Unsupported stock symbol"), 400

    # Get days from query parameter, default to 30
    days = request.args.get('days', default=30, type=int)
    if days < 1:
        days = 30

    # Fetch sentiment data
    print(f"Starting sentiment data fetch for {stock_name}...")
    sentiment_texts = fetch_sentiment_data(stock_name)
    sentiment_score, neg, neu, pos = get_sentiment_scores(sentiment_texts)

    # Fetch tweets for today
    tweets = fetch_tweets(stock_name)
    news = fetch_news_data(stock_name)

    # Calculate tweet sentiment
    tweet_compound = 0
    tweet_neg = 0
    tweet_neu = 0
    tweet_pos = 0
    if tweets and isinstance(tweets, list) and any(t.get('sentiment') for t in tweets):
        tweet_compounds = [t['sentiment']['compound'] for t in tweets if t.get('sentiment')]
        tweet_negs = [t['sentiment']['negative'] for t in tweets if t.get('sentiment')]
        tweet_neus = [t['sentiment']['neutral'] for t in tweets if t.get('sentiment')]
        tweet_poss = [t['sentiment']['positive'] for t in tweets if t.get('sentiment')]
        if tweet_compounds:
            tweet_compound = np.mean(tweet_compounds)
            tweet_neg = np.mean(tweet_negs)
            tweet_neu = np.mean(tweet_neus)
            tweet_pos = np.mean(tweet_poss)
            print(f"Tweet sentiment - compound: {tweet_compound:.4f}, neg: {tweet_neg:.4f}, neu: {tweet_neu:.4f}, pos: {tweet_pos:.4f}")

    # Calculate news sentiment
    news_compound = 0
    news_neg = 0
    news_neu = 0
    news_pos = 0
    if news and isinstance(news, list) and any(n.get('sentiment') for n in news):
        news_compounds = [n['sentiment']['compound'] for n in news if n.get('sentiment')]
        news_negs = [n['sentiment']['negative'] for n in news if n.get('sentiment')]
        news_neus = [n['sentiment']['neutral'] for n in news if n.get('sentiment')]
        news_poss = [n['sentiment']['positive'] for n in news if n.get('sentiment')]
        if news_compounds:
            news_compound = np.mean(news_compounds)
            news_neg = np.mean(news_negs)
            news_neu = np.mean(news_neus)
            news_pos = np.mean(news_poss)
            print(f"News sentiment - compound: {news_compound:.4f}, neg: {news_neg:.4f}, neu: {news_neu:.4f}, pos: {news_pos:.4f}")

    # Fallback logic: Use news sentiment if tweet sentiment is all zeros
    if tweet_compound == 0 and tweet_neg == 0 and tweet_neu == 0 and tweet_pos == 0 and (news_compound != 0 or news_neg != 0 or news_neu != 0 or news_pos != 0):
        final_compound = news_compound
        final_neg = news_neg
        final_neu = news_neu
        final_pos = news_pos
    

    # Fetch technical indicators
    print(f"Fetching technical indicators for {stock_name}...")
    indicators = get_technical_indicators(stock_name)
    if indicators is None:
        return render_template('predict.html', error="Failed to fetch technical indicators"), 500

    # Get historical data for graph
    historical_data = get_historical_data(stock_name, days)

    model, scaler_X, scaler_y = train_model(stock_name)

    try:
        full_feature_vector = [
            indicators['MA7'], indicators['MA20'], indicators['MA10'], indicators['MACD'],
            indicators['20SD'], indicators['upper_band'], indicators['lower_band'],
            indicators['EMA'], indicators['logmomentum'],
            final_compound, final_neg, final_neu, final_pos
        ]

        actual_close = indicators['Close']
        features_scaled = scaler_X.transform(np.array(full_feature_vector).reshape(1, -1))
        
        print("Making prediction with model...")
        scaled_prediction = model.predict(np.expand_dims(features_scaled, axis=0), verbose=0)[0][0]
        prediction = scaler_y.inverse_transform([[scaled_prediction]])[0][0]
        print(f"Tomorrow's prediction: {prediction:.2f}, Actual close: {actual_close:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('predict.html', error=str(e)), 500

    elapsed_time = time.time() - start_time
    print(f"Prediction completed in {elapsed_time:.2f} seconds")

    response = {
        "stock": stock_name,
        "tmrw_predicted_price": round(prediction, 2),
        "actual_close": round(actual_close, 2),
        "predicted_date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        "technical_indicators": {k: round(v, 4) if isinstance(v, float) else v for k, v in indicators.items()},
        "sentiment_score": {
            "compound": round(final_compound, 4),
            "negative": round(final_neg, 4),
            "neutral": round(final_neu, 4),
            "positive": round(final_pos, 4)
        },
        "processing_time_seconds": round(elapsed_time, 2),
        "tweets": tweets,
        "news": news,
        "historical_data": historical_data,
        "days": days
    }
    
    return render_template('predict.html', **response)

if __name__ == '__main__':
    app.run(debug=False)

