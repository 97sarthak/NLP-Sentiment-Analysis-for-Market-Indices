# NLP-Sentiment-Analysis-for-Market-Indices
## Introduction
One of the biggest factors influencing the markets is Public Sentiment. Social media platforms, especially Twitter (X), provide a wealth of real-time data that can be analyzed to gauge public opinion. This project aims to harness this data to predict stock market movements by:

1. Fetching tweets related to specific stock indices.
2. Performing sentiment analysis on these tweets using VADER (explaned in detail below).
3. Fetching historical stock data for the indices.
4. Building a predictive model to correlate sentiment with stock performance.
5. Visualizing the results.

## Project Overview

This project leverages Natural Language Processing (NLP), specifically VADER (Valence Aware Dictionary and sEntiment Reasoner), to analyze the sentiment of tweets related to stock indices. The objective is to predict the performance of major stock indices such as $QQQ by correlating the sentiment of tweets with historical stock data.

## Libraries Used 

1. tweepy: For accessing the Twitter API and fetching tweets.
2. pandas: data manipulation and analysis.
3. vaderSentiment: performing sentiment analysis on tweets.
4. yfinance: fetching historical stock data.
5. scikit-learn: building and evaluating predictive models.
6. matplotlib: visualising the results.

## What is VADER?

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically designed for text on social media. It is capable of analyzing the sentiment of text input and determining whether it is positive, negative, or neutral. Here’s how VADER works:

1. VADER uses a dictionary of words (lexicon) where each word is associated with a sentiment score. These scores are derived from human-annotated ratings of sentiment intensity.
2. VADER applies rules to handle common linguistic phenomena in social media text, such as:
    1. Punctuation: Exclamation points, for instance, amplify the sentiment intensity.
    2. Capitalization: Uppercase words are treated as having stronger sentiment.
    3. Degree Modifiers: Words like "very", "slightly", and "extremely" alter the sentiment intensity of the words they modify.
    4. Negations: Words like "not" and "never" can invert the sentiment of the words they precede.
3. VADER can handle context-sensitive sentiments, understanding the impact of conjunctions like "but" that can shift the overall sentiment of a sentence.
4. It is optimised for analyzing social media text, making it both fast and accurate, which is ideal for real-time applications.

VADER Output: Vader produces 4 sentiment metrics:
  1. Positive: The proportion of text that is positive.
  2. Negative: The proportion of text that is negative.
  3. Neutral: The proportion of text that is neutral.
  4. Compound: A normalized, weighted composite score that ranges from -1 (most extreme negative) to +1 (most extreme positive).

In this project, the compound score is used to gauge the overall sentiment of tweets.

## Code Explanation

### 1. Authenticate with Twitter API

We use the tweepy library to authenticate and access the Twitter API.

```
import tweepy

api_key = 'YOUR_API_KEY'
api_secret_key = 'YOUR_API_SECRET_KEY'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
```

### 2. Fetching Tweets

We define a function to fetch tweets related to a specific keyword.

```
import pandas as pd

def fetch_tweets(keyword, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(count)
    tweet_list = [[tweet.created_at, tweet.text] for tweet in tweets]
    df = pd.DataFrame(tweet_list, columns=['Date', 'Tweet'])
    return df
```

### 3. Performing Sentiment Analysis

Using VADER to analyse the sentiment of each tweet.

```
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df
```

### 4. Aggregate the Sentiment Scores by Date

```
tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.date
sentiment_by_date = tweets_df.groupby('Date').mean()
```
### 5. Fetch Historical Stock Data

Use yfinance to fetch historical stock data

```
import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock
```

### 6. Merge Sentiment with Data

```
merged_data = pd.merge(stock_data, sentiment_by_date, left_index=True, right_index=True, how='inner')
```

### 7. Build Predictive Model

Use scikit-learn to build a predictive model.

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare data
X = merged_data[['Sentiment']]
y = merged_data['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

### 8. Visualise the Results

```
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
```

## Results

The script will print the Mean Squared Error (MSE) of the model and display a plot comparing the actual stock prices with the predicted prices.

## Areas for Improvement

1. Fetch a larger volume of tweets for a more robust analysis. This can be done by extending the time period or increasing the number of tweets fetched per day.
2. Use advanced filtering techniques to ensure the tweets are highly relevant to the stock indices. This can include using more sophisticated keyword searches or employing machine learning models to classify relevant tweets.
3. Integrate data from other social media platforms and news sources to enrich the sentiment analysis dataset.
4. Use more advanced NLP models like BERT or RoBERTa for sentiment analysis. These models can capture more nuanced sentiment and context compared to VADER.
5. Include additional features such as trading volume, market volatility indices (VIX), and macroeconomic indicators (interest rates, unemployment rates).
6. Time Series Analysis: Implement time series forecasting methods like ARIMA, LSTM, or Prophet to model stock price movements more accurately.
7. 
