import matplotlib.pyplot as plt
import pandas as pd
import talib as ta
from processdata import *

df = pd.read_csv("SP500-spy-data.csv")
#sets "Date" as the index
df.set_index("Date", inplace=True)

def ma():
    # moving averages (10,30 and 100 days) - technical indicators
    df["MA_10"] = df["Adj Close"].rolling(window=10, min_periods=0).mean()
    df["MA_30"] = df["Adj Close"].rolling(window=30, min_periods=0).mean()
    df["MA_100"] = df["Adj Close"].rolling(window=100, min_periods=0).mean()
    # drop all NaN values
    df.dropna(inplace=True)
    print(df.head(5))
    print(df.columns)
    print(df.index)

    # moving averages - analysis
    df["Adj Close"].plot(figsize=(10, 5), title="Moving average", color="blue")
    plt.plot(df["MA_10"], label="10 day ma", color="red")
    plt.plot(df["MA_30"], label="30 day ma", color="green")
    plt.plot(df["MA_100"], label="100 day ma", color="yellow")
    plt.xlabel("Date")
    plt.ylabel("Adjusted closed price")
    plt.legend()
    plt.show()

def sma():
    # simple moving averages (10,30 and 100 days) - technical indicators
    df["SMA_10"] = df["Adj Close"].rolling(window=10, min_periods=0).mean()
    df["SMA_30"] = df["Adj Close"].rolling(window=30, min_periods=0).mean()
    df["SMA_100"] = df["Adj Close"].rolling(window=100, min_periods=0).mean()
    df.dropna(inplace=True)
    print(df.head(5))
    print(df.columns)
    print(df.index)

    # simple moving averages - analysis
    df["Adj Close"].plot(figsize=(10, 5), title="Simple Moving average", color="blue")
    plt.plot(df["SMA_10"], label="10 day sma", color="red")
    plt.plot(df["SMA_30"], label="30 day sma", color="green")
    plt.plot(df["SMA_100"], label="100 day sma", color="yellow")
    plt.xlabel("Date")
    plt.ylabel("Adjusted closed price")
    plt.legend()
    plt.show()

def ema():
    # exponential moving averages (10,30 and 100 days) - technical indicators
    df["EMA_10"] = df["Adj Close"].ewm(span=10).mean().fillna(0)
    df["EMA_30"] = df["Adj Close"].ewm(span=30).mean().fillna(0)
    df["EMA_100"] = df["Adj Close"].ewm(span=100).mean().fillna(0)
    df.dropna(inplace=True)
    print(df.head(5))
    print(df.columns)
    print(df.index)

    # exponential moving averages - analysis
    df["Adj Close"].plot(figsize=(10, 5), title="Exponential Moving average", color="blue")
    plt.plot(df["EMA_10"], label="10 day ema", color="red")
    plt.plot(df["EMA_30"], label="30 day ema", color="green")
    plt.plot(df["EMA_100"], label="100 day ema", color="yellow")
    plt.xlabel("Date")
    plt.ylabel("Adjusted closed price")
    plt.legend()
    plt.show()

def bollingder_bands():
    # bollingder bands - technical indicators
    bollingder_band_period = 20
    # Get the standard deviation
    std = df["Close"].rolling(window=bollingder_band_period).std()
    sma_bb = df["Close"].rolling(window=bollingder_band_period).mean()
    df["UPPER BAND"] = std + (std * 2)
    df["LOWER BAND"] = sma_bb - (std * 2)
    df.dropna(inplace=True)
    print(df.head(5))
    print(df.columns)
    print(df.index)

    # bollingder bands  - analysis
    df["Adj Close"].plot(figsize=(10, 5), title="Bollingder bands", color="blue")
    plt.plot(df["UPPER BAND"], label="10 day ema", color="yellow")
    plt.plot(df["LOWER BAND"], label="30 day ema", color="green")
    plt.xlabel("Date")
    plt.ylabel("Adjusted closed price")
    plt.legend()
    plt.show()

def macd():
    # macd and signal line - technical indicators
    # calculate short term ema
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    # calculate long term ema
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df.dropna(inplace=True)
    print(df.head(5))
    print(df.columns)
    print(df.index)

    # macd and signal line - analysis
    macd_line.plot(figsize=(10, 5), label="MACD", color="red", title="MACD vs Signal line")
    plt.plot(signal_line, label="Signal line", color="blue")
    plt.legend()
    plt.show()

def rsi():
    # Relative Strength Index(RSI) - technical indicators
    df["RSI"] = ta.RSI(df["Close"].values, timeperiod=14)
    df.dropna(inplace=True)
    print(df.head(5))
    print(df.columns)
    print(df.index)

    # Relative Strength Index(RSI) - analysis
    df["Adj Close"].plot(figsize=(10, 5), title="Relative Strength Index(RSI)", color="blue")
    plt.plot(df["RSI"], label="RSI", color="yellow")
    plt.xlabel("Date")
    plt.ylabel("Adjusted closed price")
    plt.legend()
    plt.show()