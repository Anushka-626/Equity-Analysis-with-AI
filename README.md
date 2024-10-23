"#Equity Analysis with AI" 

This research explores how adding sentiment analysis to stock market prediction models
can improve their accuracy. We focused on seven key European stock indices: DE40,
FR40, NL25, IT40, SP35, UK100, and EU50. To capture market sentiment, we used fi-
nancial news and applied sentiment analysis techniques. Our goal was to find out whether
using sentiment data enhances prediction models. The ARIMA model acted as a base-
line to compare the performance of Simple Neural Networks (SimpleNN), both with and
without sentiment indicators. By incorporating sentiment scores with traditional market
data, the SimpleNN model was able to recognize additional trends. We evaluated the
models using key error metrics like Mean Squared Error (MSE), Mean Absolute Error
(MAE), and Symmetric Mean Absolute Percentage Error (SMAPE). Our findings indi-
cate that sentiment data improves accuracy for some indices, but can also add noise to
others, showing that the impact of sentiment is not uniform across all indices.

This project is developed by:

Krati Pandya , Israr , Zeel Dobariya, Abhi Patel, Darshankumar Kachhadiya, Anushka Chettiar

Stock Market Prediction with ARIMA and Neural Networks

This repository contains code and scripts for predicting stock market trends using both traditional statistical methods (ARIMA) and machine learning models (Neural Networks). The repository also incorporates sentiment analysis from financial news articles to enhance prediction accuracy.

## Repository Structure

 *ADF_test.ipynb*: A Jupyter Notebook performing the Augmented Dickey-Fuller (ADF) test to check for stationarity in time series data, which is a necessary condition for applying ARIMA models.
  
 *ARIMA.ipynb*: Jupyter Notebook that applies the ARIMA (Auto-Regressive Integrated Moving Average) model to stock market data for trend forecasting. 

 *Data Scollection Script- REUTERS.py*: A Python script for scraping financial news articles from Reuters to collect text data, which will later be used for sentiment analysis.

*NN with Sentiment.py*: A Python script implementing a Simple Neural Network (SimpleNN) model with the inclusion of sentiment indicators from financial news to predict stock trends.

*NN without Indicator.py*: A Python script implementing a Simple Neural Network (SimpleNN) model without sentiment indicators, using only historical stock data for predictions.

 *NN without Sentiment.py*: Another variation of the SimpleNN model, excluding sentiment data as input.

 *Relevance Score Script.py*: This script computes the relevance score of news articles by analyzing the relationship between market events and news content.

*Sentiment Score Script.py*: This script calculates sentiment scores based on the financial news data, which is used as an additional input feature in the sentiment-based SimpleNN models.

 *stocks_data.ipynb*: Jupyter Notebook for data preprocessing, visualization, and exploration of stock market data.

## Data

 *Stocks Data*: The stocks_data.ipynb notebook loads, preprocesses, and visualizes stock market data from major indices including DE40, FR40, NL25, IT40, SP35, UK100, and EU50.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook (for .ipynb files)
- Python libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - statsmodels (for ARIMA)
  - tensorflow or pytorch (for Neural Network models)
  - nltk or textblob (for sentiment analysis)
  
You can install all required libraries using:

```bash
pip install -r requirements.txt
