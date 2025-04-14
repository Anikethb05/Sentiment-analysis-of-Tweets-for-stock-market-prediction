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
import asyncio
from twscrape import API, gather

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

def get_tech_ind(data):
    data = data.copy()

    # Ensure Close and Open exist
    if 'Close' not in data.columns or 'Open' not in data.columns:
        raise ValueError("Data must contain 'Close' and 'Open' columns")

    # Simple Moving Averages
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()

    # MACD
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    ema_12 = data['Open'].ewm(span=12, adjust=False).mean()
    data['MACD'] = ema_26 - ema_12

    # Standard Deviation
    data['20SD'] = data['Close'].rolling(window=20).std()

    # Ensure 'MA20' and '20SD' are Series
    ma20 = data['MA20'].astype(float)
    sd20 = data['20SD'].astype(float)

    # Bollinger Bands
    data.loc[:, 'upper_band'] = data['MA20'] + (2 * data['20SD'])
    data.loc[:, 'lower_band'] = data['MA20'] - (2 * data['20SD'])

    # EMA
    data['EMA'] = data['Close'].ewm(com=0.5).mean()

    # Log Momentum
    data['logmomentum'] = np.log(data['Close'] / data['Close'].shift(1))

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

    # Step 1: Fetch stock data for the past year from Yahoo Finance
    print(f"Fetching stock data for {stock_name} from Yahoo Finance...")
    try:
        end_date = datetime.now()
       # start_date = end_date - timedelta(days=365) start=start_date, end=end_date
        stock_df = yf.download(stock_name, period='180d' , progress=False)
        
        if stock_df.empty:
            raise ValueError(f"No data fetched for {stock_name}")

        # Flatten multi-index columns if present
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        # Reset index to make Date a column
        stock_df = stock_df.reset_index()
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df['Stock Name'] = stock_name
        
        # Ensure required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Stock Name']
        for col in required_cols:
            if col not in stock_df.columns:
                stock_df[col] = np.nan

        stock_df = stock_df[required_cols]
        print(f"Fetched stock data shape: {stock_df.shape}")

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        raise ValueError(f"Failed to fetch stock data for {stock_name}")

    # Step 2: Compute technical indicators
    try:
        stock_df['MA7'] = stock_df['Close'].rolling(window=7).mean()
        stock_df['MA10'] = stock_df['Close'].rolling(window=10).mean()
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()

        exp1 = stock_df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = stock_df['Close'].ewm(span=26, adjust=False).mean()
        stock_df['MACD'] = exp1 - exp2

        stock_df['20SD'] = stock_df['Close'].rolling(window=20).std()
        middle_band = stock_df['Close'].rolling(window=20).mean()
        stock_df['upper_band'] = middle_band + (2 * stock_df['20SD'])
        stock_df['lower_band'] = middle_band - (2 * stock_df['20SD'])

        stock_df['EMA'] = stock_df['Close'].ewm(span=10, adjust=False).mean()
        stock_df['logmomentum'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))

        # Drop initial rows with NaNs from indicators
        stock_df = stock_df.iloc[20:].copy()
        print(f"Stock data shape after indicators: {stock_df.shape}")

    except Exception as e:
        print(f"Error computing technical indicators: {e}")
        raise ValueError(f"Failed to compute technical indicators for {stock_name}")

    # Step 3: Fetch sentiment data (tweets and news)
    print(f"Fetching tweets for {stock_name}...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        tweets = fetch_tweets(stock_name)
        
        tweet_df = pd.DataFrame([{
            'Date': t['date'],
            'Tweet': t['text'],
            'Stock Name': stock_name,
            'sentiment_score': t['sentiment']['compound'],
            'Negative': t['sentiment']['negative'],
            'Neutral': t['sentiment']['neutral'],
            'Positive': t['sentiment']['positive']
        } for t in tweets if t['text'] != 'No tweets found for today' and t['text'] != 'Error fetching tweets'])
        
        if not tweet_df.empty:
            tweet_df['Date'] = pd.to_datetime(tweet_df['Date'])
            print(f"Fetched {len(tweet_df)} tweets")
        else:
            print("No tweets fetched")
            tweet_df = pd.DataFrame(columns=['Date', 'sentiment_score', 'Negative', 'Neutral', 'Positive'])

    except Exception as e:
        print(f"Error fetching tweets: {e}")
        tweet_df = pd.DataFrame(columns=['Date', 'sentiment_score', 'Negative', 'Neutral', 'Positive'])

    print(f"Fetching news for {stock_name}...")
    try:
        news = fetch_news_data(stock_name)
        
        news_df = pd.DataFrame([{
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Text': n['text'],
            'sentiment_score': n['sentiment']['compound'],
            'Negative': n['sentiment']['negative'],
            'Neutral': n['sentiment']['neutral'],
            'Positive': n['sentiment']['positive']
        } for n in news if n['text']])
        
        if not news_df.empty:
            news_df['Date'] = pd.to_datetime(news_df['Date'])
            print(f"Fetched {len(news_df)} news items")
        else:
            print("No news fetched")
            news_df = pd.DataFrame(columns=['Date', 'sentiment_score', 'Negative', 'Neutral', 'Positive'])

    except Exception as e:
        print(f"Error fetching news: {e}")
        news_df = pd.DataFrame(columns=['Date', 'sentiment_score', 'Negative', 'Neutral', 'Positive'])

    # Step 4: Combine sentiment data
    sentiment_df = pd.concat([tweet_df, news_df], ignore_index=True)
    if not sentiment_df.empty:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
        daily_sentiment = sentiment_df.groupby('Date').mean(numeric_only=True).reset_index()
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        print(f"Combined sentiment data shape: {daily_sentiment.shape}")
    else:
        print("No sentiment data available")
        daily_sentiment = pd.DataFrame(columns=['Date', 'sentiment_score', 'Negative', 'Neutral', 'Positive'])

    # Step 5: Merge stock data with sentiment data
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date']).dt.date
    stock_df = stock_df.merge(daily_sentiment, on='Date', how='left')

    # Fill missing sentiment values with neutral defaults
    stock_df[['sentiment_score', 'Negative', 'Neutral', 'Positive']] = stock_df[['sentiment_score', 'Negative', 'Neutral', 'Positive']].fillna({
        'sentiment_score': 0,
        'Negative': 0,
        'Neutral': 1,
        'Positive': 0
    })

    # Drop unnecessary columns
    if 'Adj Close' in stock_df.columns:
        stock_df = stock_df.drop(columns=['Adj Close'])

    # Drop any remaining NaN values
    stock_df = stock_df.dropna()
    print(f"Final data shape after merging: {stock_df.shape}")

    if stock_df.empty:
        raise ValueError(f"No valid data for {stock_name} after preprocessing")

    # Step 6: Prepare features and target
    features_list = ['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
                     'EMA', 'logmomentum', 'sentiment_score', 'Negative', 'Neutral', 'Positive']
    target = 'Close'

    for feature in features_list + [target]:
        if feature not in stock_df.columns:
            raise ValueError(f"Required feature '{feature}' not found in dataset")

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(stock_df[features_list])

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(stock_df[[target]])

    scaled_data = np.hstack((X_scaled, y_scaled))

    seq_length = 10
    X, y = create_sequences(scaled_data, seq_length)
    
    if len(X) == 0:
        raise ValueError(f"No sequences could be created for {stock_name}. Not enough data points after preprocessing.")
    
    print(f"Created {len(X)} sequences for training")

    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = 1

    # Step 7: Build and train model
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(input_dim, feature_size)),
        LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, 
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        LSTM(units=64, recurrent_dropout=0.2, 
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        Dense(32, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(16, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
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
        
        articles = soup.select('li.js-stream-content')
        for article in articles[:15]:
            headline = article.find('h3') or article.find('a')
            if headline and headline.text.strip():
                text = headline.text.strip()
                link = headline.get('href', '') if headline.name == 'a' else ''
                sentiment = analyzer.polarity_scores(text)
                news_items.append({
                    'text': text,
                    'link': link,
                    'sentiment': {
                        'compound': round(sentiment['compound'], 4),
                        'negative': round(sentiment['neg'], 4),
                        'neutral': round(sentiment['neu'], 4),
                        'positive': round(sentiment['pos'], 4)
                    }
                })
        
        if not news_items:
            articles = soup.select('h3')
            for article in articles[:15]:
                text = article.text.strip()
                if text:
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
            relevant_sentences = [s.strip() for s in all_text.split('.') if ticker in s and len(s.strip()) > 30]
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
    """Fetch sentiment data using both tweets and news."""
    tweets = fetch_tweets(stock_name)
    news = fetch_news_data(stock_name)
    
    # Combine tweet texts and news texts
    tweet_texts = [t['text'] for t in tweets if 'text' in t]
    news_texts = [n['text'] for n in news if 'text' in n]
    
    combined_texts = tweet_texts + news_texts
    print(f"Combined {len(tweet_texts)} tweets and {len(news_texts)} news items for sentiment analysis")
    
    if not combined_texts:
        print(f"Warning: No sentiment data for {stock_name}")
        return []
        
    return combined_texts

def get_sentiment_scores(texts):
    """Calculate sentiment scores with improved error handling."""
    if not texts:
        print("Warning: No texts available for sentiment analysis")
        return 0,0,1,0
    
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
            return 0,0,1,0
            
        sentiment_score = np.mean(compound_scores)
        positive = np.mean(positives)
        neutral = np.mean(neutrals)
        negative = np.mean(negatives)
        
        print(f"Sentiment analysis complete - compound: {sentiment_score:.4f}, neg: {negative:.4f}, neu: {neutral:.4f}, pos: {positive:.4f}")
        return sentiment_score, negative, neutral, positive

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0,0,1,0

def fetch_tweets(stock_name):
    try:
        stock_name = stock_name.upper()
        print(f"Fetching tweets for {stock_name} for today...")
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        start_str = today.strftime('%Y-%m-%d')
        end_str = tomorrow.strftime('%Y-%m-%d')
        
        api = API()
        query = f"${stock_name} OR #{stock_name} OR {stock_name} stock lang:en since:{start_str} until:{end_str}"
        tweets = []
        
        async def search_tweets():
            async for tweet in api.search(query, limit=50):
                if tweet.date.date() == today:
                    sentiment = analyzer.polarity_scores(tweet.rawContent)
                    tweets.append({
                        'text': tweet.rawContent,
                        'link': f"https://twitter.com/i/status/{tweet.id}",
                        'sentiment': {
                            'compound': round(sentiment['compound'], 4),
                            'negative': round(sentiment['neg'], 4),
                            'neutral': round(sentiment['neu'], 4),
                            'positive': round(sentiment['pos'], 4)
                        },
                        'date': tweet.date.strftime('%Y-%m-%d')
                    })
        
        asyncio.run(search_tweets())
        
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

    # Fetch sentiment data (tweets and news combined)
    print(f"Starting sentiment data fetch for {stock_name}...")
    sentiment_texts = fetch_sentiment_data(stock_name)
    sentiment_score, neg, neu, pos = get_sentiment_scores(sentiment_texts)

    # Fetch tweets and news for display
    tweets = fetch_tweets(stock_name)
    news = fetch_news_data(stock_name)

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
            sentiment_score, neg, neu, pos
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
            "compound": round(sentiment_score, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
            "positive": round(pos, 4)
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
