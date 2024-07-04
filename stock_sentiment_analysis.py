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

def fetch_tweets(keyword, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(count)
    tweet_list = [[tweet.created_at, tweet.text] for tweet in tweets]
    df = pd.DataFrame(tweet_list, columns=['Date', 'Tweet'])
    return df

def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

# Fetch and process tweets
tweets_df = fetch_tweets('$QQQ', count=1000)
tweets_df = analyze_sentiment(tweets_df)
tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.date
sentiment_by_date = tweets_df.groupby('Date').mean()

# Fetch historical stock data
stock_data = fetch_stock_data('QQQ', '2023-01-01', '2024-01-01')

# Merge sentiment with stock data
merged_data = pd.merge(stock_data, sentiment_by_date, left_index=True, right_index=True, how='inner')

# Prepare data for modeling
X = merged_data[['Sentiment']]
y = merged_data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
