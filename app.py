import time
from functools import lru_cache
import yfinance as yf
import pandas_ta as ta
from flask import Flask, jsonify
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


# Load model and scalers
model = load_model("saved_model/lstm_stock_model.h5")

with open("saved_model/scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("saved_model/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

analyzer = SentimentIntensityAnalyzer()
app = Flask(__name__)

# Supported stocks
STOCKS = ['TSLA', 'MSFT', 'PG', 'META', 'AMZN', 'GOOG', 'AMD', 'AAPL',
          'NFLX', 'TSM', 'KO', 'F', 'COST', 'DIS', 'VZ', 'CRM', 'INTC', 'BA',
          'BX', 'NOC', 'PYPL', 'ENPH', 'NIO', 'ZS', 'XPEV']

FEATURE_TEMPLATE = ['MA7', 'MA20', 'MA10', 'MACD', '20SD', 'upper_band', 'lower_band',
                    'EMA', 'logmomentum', 'sentiment_score', 'Negative', 'Neutral', 'Positive']


def fetch_news_data(ticker):
    """Fetch stock news from Yahoo Finance as an alternative to Twitter."""
    try:
        print(f"Fetching Yahoo Finance news for {ticker}")
        # User agent to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Yahoo Finance news URL
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        # Extract headlines and descriptions
        articles = soup.find_all("div", {"class": "Ov(h) Pend(44px) Pstart(25px)"})
        if not articles:
            # Try alternative class names
            articles = soup.find_all("h3", {"class": "Mb(5px)"})
        
        for article in articles[:15]:  # Limit to 15 articles
            if hasattr(article, 'text') and article.text:
                news_items.append(article.text.strip())
        
        # If we didn't find news with the HTML parsing approach
        if not news_items:
            # Try to find any news-related text
            all_text = soup.get_text()
            # Extract paragraphs that mention the ticker
            relevant_sentences = [s.strip() for s in all_text.split('.') 
                                 if ticker in s and len(s) > 30]
            news_items.extend(relevant_sentences[:15])
        
        print(f"Found {len(news_items)} news items for {ticker}")
        if news_items:
            for i, item in enumerate(news_items[:3]):
                print(f"- News {i+1}: {item[:100]}...")
                
        return news_items
        
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


def get_technical_indicators(symbol):
    try:
        # First try 60 days
        df = yf.download(symbol, period='60d', interval='1d')
        
        # If not enough data, try 90 days
        if len(df) < 26:  # Need at least 26 days for MACD calculation
            print(f"Not enough data with 60 days, trying 90 days for {symbol}")
            df = yf.download(symbol, period='90d', interval='1d')
            
        if len(df) < 26:
            raise ValueError(f"Not enough data points for {symbol}, only got {len(df)}")

        features = {}
        features['MA7'] = float(df['Close'].rolling(window=7).mean().iloc[-1])
        features['MA10'] = float(df['Close'].rolling(window=10).mean().iloc[-1])
        features['MA20'] = float(df['Close'].rolling(window=20).mean().iloc[-1])

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        features['MACD'] = float(macd_line.iloc[-1])

        features['20SD'] = float(df['Close'].rolling(window=20).std().iloc[-1])

        middle_band = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        features['upper_band'] = float((middle_band + (std_dev * 2)).iloc[-1])
        features['lower_band'] = float((middle_band - (std_dev * 2)).iloc[-1])

        features['EMA'] = float(df['Close'].ewm(span=10, adjust=False).mean().iloc[-1])

        current_close = float(df['Close'].iloc[-1])
        past_close = float(df['Close'].iloc[-11]) if len(df) >= 11 else float(df['Close'].iloc[0])
        momentum = current_close - past_close
        features['logmomentum'] = (
            float(np.log1p(momentum)) if momentum > 0 else
            float(-np.log1p(abs(momentum))) if momentum < 0 else 0.0
        )

        features['Close'] = float(current_close)

        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in features.values()):
            missing = [k for k, v in features.items() if v is None or (isinstance(v, float) and np.isnan(v))]
            raise ValueError(f"Missing values in indicators: {missing}")

        return features

    except Exception as e:
        print(f"Error in technical indicators: {e}")
        return None


@app.route('/predict/<stock_name>')
def predict(stock_name):
    start_time = time.time()
    stock_name = stock_name.upper()
    if stock_name not in STOCKS:
        return jsonify({"error": "Unsupported stock symbol"}), 400

    # Fetch sentiment data
    print(f"Starting sentiment data fetch for {stock_name}...")
    sentiment_texts = fetch_sentiment_data(stock_name)
    print(f"Fetched {len(sentiment_texts)} text items for sentiment analysis")
    
    # Calculate sentiment
    sentiment_score, neg, neu, pos = get_sentiment_scores(sentiment_texts)
    print(f"Sentiment analysis results: compound={sentiment_score:.4f}, neg={neg:.4f}, neu={neu:.4f}, pos={pos:.4f}")

    # Get technical indicators
    print(f"Fetching technical indicators for {stock_name}...")
    indicators = get_technical_indicators(stock_name)
    if indicators is None:
        return jsonify({"error": "Failed to fetch technical indicators"}), 500
    print(f"Technical indicators fetched successfully")

    try:
        # Create feature vector excluding 'Close' for prediction
        full_feature_vector = [
            indicators['MA7'], indicators['MA20'], indicators['MA10'], indicators['MACD'],
            indicators['20SD'], indicators['upper_band'], indicators['lower_band'],
            indicators['EMA'], indicators['logmomentum'],
            sentiment_score, neg, neu, pos
        ]

        actual_close = indicators['Close']
        features_scaled = scaler_X.transform(np.array(full_feature_vector).reshape(1, -1))
        
        # Make prediction
        print("Making prediction with model...")
        scaled_prediction = model.predict(np.expand_dims(features_scaled, axis=0), verbose=0)[0][0]
        prediction = scaler_y.inverse_transform([[scaled_prediction]])[0][0]
        print(f"Raw prediction: {prediction:.2f}, Actual close: {actual_close:.2f}")

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

    elapsed_time = time.time() - start_time
    print(f"Prediction completed in {elapsed_time:.2f} seconds")

    # Prepare response
    response = {
        "stock": stock_name,
        "predicted_price": round(prediction, 2),
        "actual_close": round(actual_close, 2),
        "technical_indicators": {k: round(v, 4) if isinstance(v, float) else v for k, v in indicators.items()},
        "sentiment_score": {
            "compound": round(sentiment_score, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
            "positive": round(pos, 4)
        },
        "text_count": len(sentiment_texts),
        "processing_time_seconds": round(elapsed_time, 2)
    }
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
