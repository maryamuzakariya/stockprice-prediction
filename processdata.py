#import libraries 
import csv
import requests
import os #creates new directory source
import pickle #serialise python objects for SP500 to be in a list
import bs4 as bs
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from matplotlib import colors
from matplotlib import style
from pandas_datareader import data as pdr
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

yf.pdr_override()
style.use("ggplot")


def download_sp500_data():
    #download available data of the s&p500 index from yahoo finance
    sp500_data = yf.download("^GSPC", start="2017-01-01", end="2021-10-16")
    #creates a dataframe of the downloaded data
    sp500_df = pd.DataFrame(sp500_data)
    #saves to and creates csv file
    sp500_df.to_csv("sp500_data.csv")

def visualise_data():
    read_df = pd.read_csv("sp500_data.csv")
    #sets date as the index
    read_df.set_index("Date", inplace=True)
    read_df['Adj Close'].plot()
    plt.ylabel("Adjusted Close Prices")
    plt.show()


def visualise_sp500_companies():
    # getting and showing the correlation among the dataset, especially how they are postively and negatively correlated
    # correlation analysis
    df = pd.read_csv("SP500companies_adjclose_combined.csv")
    correlate_df = df.corr()
    print(correlate_df.head())
    #visualise the correlation data
    data_df = correlate_df.values
    fig = plt.figure()
    #i by 1, plot number 1
    ax = fig.add_subplot(1, 1, 1)
    sandp500_data_heatmap = ax.pcolor(data_df, cmap=plt.cm.hot)
    fig.colorbar(sandp500_data_heatmap)
    #arranging ticks at half marks
    ax.set_xticks(np.arange(data_df.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data_df.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    label_columns = correlate_df.columns
    label_row = correlate_df.index
    ax.set_xticklabels(label_columns)
    ax.set_yticklabels(label_row)
    plt.xticks(rotation=90)
    # set color limit
    sandp500_data_heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.title("SP500 Companies Correlation Heatmap", fontsize=18)
    plt.show()


def train_test_predict():
    #divide data into train and test. #build, fit model and predict
    #evaluate model - regression metrics
    # Train: 80%, Test: 20% ------point this for solving a performance problem 
    df = pd.read_csv("sp500_data.csv")
    #sets date as the index
    df.set_index("Date", inplace=True)
    df.dropna(inplace=True)

    #select x and y features 
    #[ x = open, high, low, close, adjclose ] & [ y = adjclose ]
    x = df.iloc[:, 0:5].values
    y = df.iloc[:, 4].values
    print(x)
    print(y)
    print(x.shape)
    print(y.shape)
    #x.shape = (1079, 5) and y.shape = (1079,)

    #divide into training and testing set
    #train: 74% and test: 26% --------point this for solving a performance problem 
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,  random_state=0) ----  initial
    #x_train.shape = (798, 5) and x_test.shape = (281, 5) and y_train.shape = (798,) and y_test.shape = (281,)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0)

    #scaling the features
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)

    #define, build, fit model and predict 
    #model = RandomForestRegressor(n_estimators=20, random_state=0) ----- initial parameters before bypertuning and after is below 
    model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    print(predict)
    print(predict.shape)

    #hypertuning parameters using the random serach cross validation 
    #creates the random grid 
    
    grid_rf = {
        'n_estimators': [20, 50, 100, 500, 1000],  # Number of Trees in the Forest
        'max_depth': np.arange(1, 15, 1),  # Maximum Depth of Each Tree - max number of levels in a tree
        'min_samples_split': [2, 10, 9],  # Minimum Number of Samples Required to Split Internal Node - minimum number of data point before the sample is split 
        'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  # Minimum Number of Samples Required in Leaf Node - minimum number of leaf node 
        'bootstrap': [True, False], # Bootstap Sample (Sampling with Replacement) True of False- bootstrap sampling for data points - true or false
        "random_state": [1, 2, 30, 42]
    }
    
    rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
    rscv_fit = rscv.fit(x_train, y_train)
    best_parameters = rscv_fit.best_params_
    print(best_parameters)

    {'random_state': 42, 'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10, 'bootstrap': True} 

    print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
    print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
    print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
    # # The best possible score is 1.0, lower values are worse.
    # print("Explained Variance Score:", metrics.explained_variance_score(y_test, predict))
    # Best possible score is 1.0
    print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
    print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
    
    errors = abs(predict - y_test)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')


    #creates a df with predictions and dates;
    predictions = pd.DataFrame({"Predictions": predict}, index=pd.date_range(start=df.index[-1], periods=len(predict), freq="D"))
    predictions.to_csv("Predicted-price-data.csv")
    print(predictions)

    # colllects future days from predicted values
    oneyear_df = pd.DataFrame(predictions[:252])
    oneyear_df.to_csv("one-year-predictions.csv")

    sixmonths_df = pd.DataFrame(predictions[:126])
    sixmonths_df.to_csv("six-months-predictions.csv")

    onemonth_df = pd.DataFrame(predictions[:21])
    onemonth_df.to_csv("one-month-predictions.csv")

    fivedays_df = pd.DataFrame(predictions[:5])
    fivedays_df.to_csv("five-days-predictions.csv")

    # plot 30 days for y_test and predict 
    plt.plot(y_test[:10], label="Expected", color="blue")
    plt.plot(predict[:10], label="Predicted", color="orange")
    plt.legend()
    plt.show()

def oneyear_prediction():
    oneyear_df_pred = pd.read_csv("one-year-predictions.csv")
    oneyear_df_pred.set_index("Date", inplace=True)
    buy_price = min(oneyear_df_pred["Predictions"])
    sell_price = max(oneyear_df_pred["Predictions"])
    oneyear_buy = oneyear_df_pred.loc[oneyear_df_pred["Predictions"] == buy_price]
    oneyear_sell = oneyear_df_pred.loc[oneyear_df_pred["Predictions"] == sell_price]
    print("Buy price and date")
    print(oneyear_buy)
    print("Sell price and date")
    print(oneyear_sell)

    oneyear_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 1 year", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def sixmonths_prediction():
    sixmonths_df_pred = pd.read_csv("six-months-predictions.csv")
    sixmonths_df_pred.set_index("Date", inplace=True)
    buy_price = min(sixmonths_df_pred["Predictions"])
    sell_price = max(sixmonths_df_pred["Predictions"])
    sixmonths_buy = sixmonths_df_pred.loc[sixmonths_df_pred["Predictions"] == buy_price]
    sixmonths_sell = sixmonths_df_pred.loc[sixmonths_df_pred["Predictions"] == sell_price]
    print("Buy price and date")
    print(sixmonths_buy)
    print("Sell price and date")
    print(sixmonths_sell)

    sixmonths_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 6 months", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def onemonth_prediction():
    onemonth_df_pred = pd.read_csv("one-month-predictions.csv")
    onemonth_df_pred.set_index("Date", inplace=True)
    buy_price = min(onemonth_df_pred["Predictions"])
    sell_price = max(onemonth_df_pred["Predictions"])
    onemonth_buy = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == buy_price]
    onemonth_sell = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == sell_price]
    print("Buy price and date")
    print(onemonth_buy)
    print("Sell price and date")
    print(onemonth_sell)

    onemonth_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 1 month", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def fivedays_prediction():
    fivedays_df_pred = pd.read_csv("five-days-predictions.csv")
    fivedays_df_pred.set_index("Date", inplace=True)
    buy_price = min(fivedays_df_pred["Predictions"])
    sell_price = max(fivedays_df_pred["Predictions"])
    fivedays_buy = fivedays_df_pred.loc[fivedays_df_pred["Predictions"] == buy_price]
    fivedays_sell = fivedays_df_pred.loc[fivedays_df_pred["Predictions"] == sell_price]
    print("Buy price and date")
    print(fivedays_buy)
    print("Sell price and date")
    print(fivedays_sell)

    fivedays_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 5 days", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    