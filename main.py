# -*- coding: utf-8 -*-
"""
Created on Mon Dec  13 11:54:56 2021

@author: Vivek Srivastava
"""
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDClassifier
import yfinance as yf

st.title('Prediction Asset Prices (+ Crypto)')
st.sidebar.title("CryptoCurrency")
option = st.sidebar.selectbox(
     'Select Crypto?',
     ("BTC-USD","ETH-USD","BNB-USD","USDT-USD","SOL1-USD","USDC-USD","ADA-USD","XRP-USD","HEX-USD","DOGE-USD","AVAX-USD","SHIB-USD","CRO-USD","MATIC-USD","LTC-USD","UNI3-USD","TRX-USD","ALGO-USD","LINK-USD","BCH-USD","DAI1-USD","XLM-USD","MANA-USD","AXS-USD","FTT1-USD","VET-USD","FIL-USD"
))

import plotly.graph_objs as go
ticker = yf.Ticker(option)
history = ticker.history(period="max")
history= history.reset_index()
history['Close'].plot(title=f"{option} Historical Price Data")
history.dropna(inplace=True)
required_features = ['Open', 'High', 'Low', 'Volume','Dividends', 'Stock Splits']
output_label = 'Close'
x_train, x_test, y_train, y_test = train_test_split(
                                                history[required_features],
                                                history[output_label],
                                                test_size = 0.3
                                                 )

model = LinearRegression()

model.fit(x_train, y_train)
future_set = history.shift(periods=30).tail(30)
prediction = model.predict(future_set[required_features])
plt.figure(figsize = (12, 7))
plt.plot(history["Date"][-400:-60], history["Close"][-400:-60], color='goldenrod', lw=2)
plt.plot(future_set["Date"], prediction, color='deeppink', lw=2)
plt.title("History Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)
st.pyplot(plt)