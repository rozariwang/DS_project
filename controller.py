from datetime import datetime, timedelta

import openai
import pandas as pd
import streamlit as st
import torch
import yahoo_fin.stock_info as si
from stocksent import Sentiment
from transformers import AutoModelForSequenceClassification, AutoTokenizer

''' :)
# Retrieve OpenAI API credentials from secrets.toml
secrets = st.secrets["openai"]
openai_organization = secrets["openai_organization"]
openai_api = secrets["openai_api"]
'''   

# Set up OpenAI API credentials
#openai.organization = openai_organization
openai.organization = "org-pJcWPQGFUTRBlstxxYtLSgys"
#openai.api_key = openai_api
openai.api_key = "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"

def get_sentiment(input_text: str) -> "list[float]":
    """
    Performs sentiment analysis on the input text using the FinBERT model.
    Args:
        input_text (str): The text for sentiment analysis.
    Returns:
        list: A list of sentiment probabilities for each class.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    return torch.nn.Softmax(dim=1)(logits)[0].tolist()

def get_stock_data(ticker: str) -> "tuple[torch.tensor(), float]": # type: ignore
    """
    Retrieves stock data, performs sentiment analysis on related news stories,
    and returns the processed data as a tuple.
    Args:
        ticker (str): The stock ticker symbol.
    Returns:
        tuple: A tuple containing the processed stock data tensor and the annual percent change.
    """
    today: datetime = datetime.today()
    yesteryear: str = (today - timedelta(days=345)).strftime('%Y-%m-%d')
    column_names: "list[str]" = ["low","open","volume","high","close","adjclose","Annual Percent Change","positive","negative","neutral"]

    stock_news = Sentiment(ticker)
    sentiment_score = stock_news.get_dataframe(days=1)
    stories = sentiment_score['headline'].tolist()[:10]
    sentiments: "list[float]" = []

    for story in stories:
        sentiment = get_sentiment(story)
        sentiments.append(sentiment)

    sentiments_df = pd.DataFrame(sentiments, columns=['positive', 'negative', 'neutral'])
    mean_sentiments: pd.Series = sentiments_df.mean()
    stock_sentiment: "list[float]" = mean_sentiments.values.tolist()

    ticker_info: pd.DataFrame = si.get_data(ticker, start_date=yesteryear, end_date=today, interval="1mo")
    ticker_info = ticker_info.drop(columns=['ticker'])
    ticker_info['Annual Percent Change'] = (ticker_info.iloc[-1]['close'] - ticker_info.iloc[0]['close']) / ticker_info.iloc[0]['close'] * 100
    annualized_ticker_info: pd.Series = ticker_info.mean()

    annualized_ticker_info['positive'] = stock_sentiment[0]
    annualized_ticker_info['negative'] = stock_sentiment[1]
    annualized_ticker_info['neutral'] = stock_sentiment[2]

    annualized_ticker_info = annualized_ticker_info[column_names] # reorder columns

    return torch.Tensor(annualized_ticker_info.values.tolist()), annualized_ticker_info['Annual Percent Change']

def generate_stock_prediction(company: str) -> "tuple[float, float]":
    """
    Generates a stock prediction for a given company using the retrieved stock data and a loaded model.
    Args:
        company (str): The name or symbol of the company.
    Returns:
        tuple: A tuple containing the stock prediction and the annual percent change.
    """
    stock_data, annual_percent_change = get_stock_data(company)
    model = torch.load('multilayer_model2.pickle') ### TODO upload whatever model Nicho finishes
    prediction = model(stock_data)
    return prediction.item(), annual_percent_change

# Function to generate recommendation using ChatGPT API
@st.cache
def generate_recommendation(company: str):
    """
    Generates a recommendation for a given company based on the stock prediction and annual percent change.
    Args:
        company (str): The name or symbol of the company.
    Returns:
        str: The generated recommendation as a response to the prompt.
    """

    companies = pd.read_csv('dicker_lookup_df.csv', index_col=0)
    companies = companies.to_dict("split")
    companies = zip(companies["index"], companies["data"])
    compnay_dict = {k: v[0] for k, v in companies}

    company_ticker = compnay_dict[company]

    prediction, annual_percent_change = generate_stock_prediction(company_ticker)
    
    prompt = str(f"""Given the score -1(not promising) to 1(very promosing), 
              our model gives {company_ticker} company a score of {prediction}. 
              Based on this score {prediction}, and an annual growth of {annual_percent_change}, 
              explain why {company_ticker} company stock might or might not be suitable to invest in the long term? 
              Respond in less than 150 words.""")
    
    completion = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.6
    )
    response = completion.choices[0].text
    return response
