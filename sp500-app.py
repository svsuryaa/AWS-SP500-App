import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

# Disable the warning about Matplotlib's global figure object
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df

df = load_data()
sector = df.groupby('GICS Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted(df['GICS Sector'].unique())
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data
df_selected_sector = df[df['GICS Sector'].isin(selected_sector)]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)

# Download S&P500 data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# Fetch stock data using yfinance
data = yf.download(
    tickers=list(df_selected_sector[:10].Symbol),
    period="ytd",
    interval="1d",
    group_by='ticker',
    auto_adjust=True,
    prepost=True,
    threads=True,
    proxy=None
)

# Plot Closing Price of Query Symbol
def price_plot(symbol):
    df = pd.DataFrame(data[symbol].Close)
    df['Date'] = df.index
    fig, ax = plt.subplots()  # Create a new figure and axis for each plot
    ax.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
    ax.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
    ax.set_xticks(df.Date[::30])  # Set x-axis ticks every 30 days
    ax.set_title(symbol, fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Closing Price(in US$)', fontweight='bold')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    return st.pyplot(fig)  # Pass the figure explicitly to st.pyplot()

# Historical Stock Price Plot
def historical_price_plot(symbol):
    hist_data = yf.download(symbol, period="1y")
    plt.figure(figsize=(10, 6))
    plt.plot(hist_data['Close'])
    plt.title(f'Historical Stock Price of {symbol}', fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Price (in US$)', fontweight='bold')
    st.pyplot()

# Moving Average Plot
def moving_average_plot(symbol):
    hist_data = yf.download(symbol, period="1y")
    hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
    hist_data['MA200'] = hist_data['Close'].rolling(window=200).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(hist_data['Close'], label='Close Price')
    plt.plot(hist_data['MA50'], label='50-Day Moving Average')
    plt.plot(hist_data['MA200'], label='200-Day Moving Average')
    plt.title(f'Moving Averages for {symbol}', fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Price (in US$)', fontweight='bold')
    plt.legend()
    st.pyplot()

num_company = st.sidebar.slider('Number of Companies', 1, 5)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)
        historical_price_plot(i)
        moving_average_plot(i)
