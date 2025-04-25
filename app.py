import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Nifty 50 - Last 1 Year Data")
nifty = yf.Ticker("^NSEI")
data = nifty.history(period="1y")
st.dataframe(data)
st.line_chart(data['Close'])
