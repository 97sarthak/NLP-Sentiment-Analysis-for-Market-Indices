import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Twitter API credentials
api_key = 'YOUR_API_KEY'
api_secret_key = 'YOUR_API_SECRET_KEY'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def fetch_tweets(keywords, count=100):
    all_tweets = []
    for keyword in keywords:
        tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(count)
        tweet_list = [[tweet.created_at, tweet.text, keyword] for tweet in tweets]
        all_tweets.extend(tweet_list)
    df = pd.DataFrame(all_tweets, columns=['Date', 'Tweet', 'Keyword'])
    return df

def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df

def fetch_stock_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Keywords for various stock indices and related terms
keywords = [
    '$QQQ', '$SPY', '$DIA', '$IWM', '$FTSE', '$DAX', 
    '#stockmarket', '#investing', '#trading', '#finance', 
    '#NASDAQ', '#S&P500', '#DowJones', '#stocks', '#equities'
]

# Fetch and process tweets
tweets_df = fetch_tweets(keywords, count=500)
tweets_df = analyze_sentiment(tweets_df)
tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.date
sentiment_by_date_keyword = tweets_df.groupby(['Date', 'Keyword']).mean().reset_index()

# Stock tickers for corresponding keywords
tickers = {
    '$QQQ': 'QQQ',
    '$SPY': 'SPY',
    '$DIA': 'DIA',
    '$IWM': 'IWM',
    '$FTSE': '^FTSE',
    '$DAX': '^GDAXI'
}

# Fetch historical stock data
stock_data = fetch_stock_data(tickers.values(), '2023-01-01', '2024-01-01')

# Merge sentiment with stock data for each ticker
merged_data = {}
for keyword, ticker in tickers.items():
    if ticker in stock_data:
        sentiment_data = sentiment_by_date_keyword[sentiment_by_date_keyword['Keyword'] == keyword]
        stock_df = stock_data[ticker]
        stock_df['Date'] = stock_df.index.date
        merged_df = pd.merge(stock_df, sentiment_data, on='Date', how='inner')
        merged_data[ticker] = merged_df

# Prepare data for modeling and visualization
for ticker, data in merged_data.items():
    if not data.empty:
        X = data[['Sentiment']]
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        print(f'{ticker} - Mean Squared Error: {mse}')

        # Visualize the results
        plt.figure(figsize=(14, 7))
        plt.plot(y_test.index, y_test, label='Actual')
        plt.plot(y_test.index, y_pred, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.legend()
        plt.show()

