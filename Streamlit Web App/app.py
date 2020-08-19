
# coding: utf-8

# In[8]:



# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import base64
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import time

from PIL import Image
import plotly.graph_objects as go


from datetime import datetime
import os 
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import talib

from typing import TypeVar, Callable, Sequence
from functools import reduce
T = TypeVar('T')



import glob
from IPython.display import display, HTML
from tqdm import tqdm_notebook as tqdm
from sklearn import preprocessing

import json


import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))



import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import talib

from typing import TypeVar, Callable, Sequence
from functools import reduce
T = TypeVar('T')

class Stocks:
    def __init__(self, ticker, start_date, forcast_horz):
        self.Ticker = ticker
        self.Start_Date = start_date
        self.forcast_horz = forcast_horz
        
    
    def get_stock_data(self, Ticker):
    
        #ALPHA_VANTAGE_API_KEY = 'XAGC5LBB1SI9RDLW'
        #self.ts = TimeSeries(key= ALPHA_VANTAGE_API_KEY, output_format='pandas')
        print('Loading Historical Price data for ' + self.Ticker + '....')
        #self.df_Stock, self.Stock_info = self.ts.get_daily(self.Ticker, outputsize='full') 
        Stock_obj = yf.Ticker(self.Ticker)
        self.df_Stock = Stock_obj.history(start=self.Start_Date)
        print(self.df_Stock)
        #print(self.Stock_info)
        #self.df_Stock = self.df_Stock.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'Close', '5. volume': 'Volume' })
        #self.df_Stock = self.df_Stock.rename_axis(['Date'])
        #sorting index
        self.Stock = self.df_Stock.sort_index(ascending=True, axis=0)
        self.Stock = self.Stock.drop(columns=['Dividends', 'Stock Splits'])
        print(self.Stock)
        #slicing the data for 15 years from '2004-01-02' to today
        #self.Stock = self.Stock.loc[self.Start_Date:]

        fig = self.Stock[['Close', 'High']].plot()
        plt.title("Stock Price Over time", fontsize=17)
        plt.ylabel('Price', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        #plt.show()
        #st.pyplot(fig)

  
    def extract_Technical_Indicators(self, Ticker):
        
        print(' ')
        print('Feature extraction of technical Indicators....')
        #get Boolinger Bands
        self.Stock['MA_20'] = self.Stock.Close.rolling(window=20).mean()
        self.Stock['SD20'] = self.Stock.Close.rolling(window=20).std()
        self.Stock['Upper_Band'] = self.Stock.Close.rolling(window=20).mean() + (self.Stock['SD20']*2)
        self.Stock['Lower_Band'] = self.Stock.Close.rolling(window=20).mean() - (self.Stock['SD20']*2)
        print('Boolinger bands..')

        print(self.Stock.shape)
        #shifting for lagged data 
        self.Stock['S_Close(t-1)'] = self.Stock.Close.shift(periods=1)
        self.Stock['S_Close(t-2)'] = self.Stock.Close.shift(periods=2)
        self.Stock['S_Close(t-3)'] = self.Stock.Close.shift(periods=3)
        self.Stock['S_Close(t-5)'] = self.Stock.Close.shift(periods=5)
        self.Stock['S_Open(t-1)'] = self.Stock.Open.shift(periods=1)
        print('Lagged Price data for previous days....')

        #simple moving average
        self.Stock['MA5'] = self.Stock.Close.rolling(window=5).mean()
        self.Stock['MA10'] = self.Stock.Close.rolling(window=10).mean()
        self.Stock['MA20'] = self.Stock.Close.rolling(window=20).mean()
        self.Stock['MA50'] = self.Stock.Close.rolling(window=50).mean()
        self.Stock['MA200'] = self.Stock.Close.rolling(window=200).mean()
        print('Simple Moving Average....')

        #Exponential Moving Averages
        self.Stock['EMA10'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
        self.Stock['EMA20'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
        self.Stock['EMA50'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
        self.Stock['EMA100'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
        self.Stock['EMA200'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
        print('Exponential Moving Average....')

        #Moving Average Convergance Divergances
        self.Stock['EMA_12'] = self.Stock.Close.ewm(span=12, adjust=False).mean()
        self.Stock['EMA_26'] = self.Stock.Close.ewm(span=26, adjust=False).mean()
        self.Stock['MACD'] = self.Stock['EMA_12'] - self.Stock['EMA_26']

        self.Stock['MACD_EMA'] = self.Stock.MACD.ewm(span=9, adjust=False).mean()

        #Average True Range
        self.Stock['ATR'] = talib.ATR(self.Stock['High'].values, self.Stock['Low'].values, self.Stock['Close'].values, timeperiod=14)

        #Average Directional Index
        self.Stock['ADX'] = talib.ADX(self.Stock['High'], self.Stock['Low'], self.Stock['Close'], timeperiod=14)

        #Commodity Channel index
        tp = (self.Stock['High'] + self.Stock['Low'] + self.Stock['Close']) /3
        ma = tp/20 
        md = (tp-ma)/20
        self.Stock['CCI'] = (tp-ma)/(0.015 * md)
        print('Commodity Channel Index....')

        #Rate of Change 
        self.Stock['ROC'] = ((self.Stock['Close'] - self.Stock['Close'].shift(10)) / (self.Stock['Close'].shift(10)))*100

        #Relative Strength Index
        self.Stock['RSI'] = talib.RSI(self.Stock.Close.values, timeperiod=14)

        #William's %R
        self.Stock['William%R'] = talib.WILLR(self.Stock.High.values, self.Stock.Low.values, self.Stock.Close.values, 14) 

        #Stocastic K
        self.Stock['SO%K'] = ((self.Stock.Close - self.Stock.Low.rolling(window=14).min()) / (self.Stock.High.rolling(window=14).max() - self.Stock.Low.rolling(window=14).min())) * 100
        print('Stocastic %K ....')
        #Standard Deviation of last 5 day returns
        self.Stock['per_change'] = self.Stock.Close.pct_change()
        self.Stock['STD5'] = self.Stock.per_change.rolling(window=5).std()

        #Force Index
        self.Stock['ForceIndex1'] = self.Stock.Close.diff(1) * self.Stock.Volume
        self.Stock['ForceIndex20'] = self.Stock.Close.diff(20) * self.Stock.Volume
        print('Force index....')

        #print('Stock Data ', self.Stock)
        
        self.Stock[['Close', 'MA_20', 'Upper_Band', 'Lower_Band']].plot(figsize=(12,6))
        plt.title('20 Day Bollinger Band')
        plt.ylabel('Price (USD)')
        plt.show();
        #st.pyplot(fig1)
        
        self.Stock[['Close', 'MA20', 'MA200', 'MA50']].plot()
        plt.show();

        self.Stock[['MACD', 'MACD_EMA']].plot()
        plt.show();
        #st.pyplot(fig2)
        #Dropping unneccesary columns
        self.Stock = self.Stock.drop(columns=['MA_20', 'per_change', 'EMA_12', 'EMA_26'])
        print(self.Stock.shape)

        
    def extract_info(self, date_val):

        Day = date_val.day
        DayofWeek = date_val.dayofweek
        Dayofyear = date_val.dayofyear
        Week = date_val.week
        Is_month_end = date_val.is_month_end.real
        Is_month_start = date_val.is_month_start.real
        Is_quarter_end = date_val.is_quarter_end.real
        Is_quarter_start = date_val.is_quarter_start.real
        Is_year_end = date_val.is_year_end.real
        Is_year_start = date_val.is_year_start.real
        Is_leap_year = date_val.is_leap_year.real
        Year = date_val.year
        Month = date_val.month
        
        return Day, DayofWeek, Dayofyear, Week, Is_month_end, Is_month_start, Is_quarter_end, Is_quarter_start, Is_year_end, Is_year_start, Is_leap_year, Year, Month


    def extract_date_features(self, Ticker):
        print(' ')
        
        self.Stock['Date_col'] = self.Stock.index
        
        self.Stock[['Day', 'DayofWeek', 'DayofYear', 'Week', 'Is_month_end', 'Is_month_start',
          'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Is_leap_year', 'Year', 'Month']] = self.Stock.Date_col.apply(lambda date_val: pd.Series(self.extract_info(date_val)))
        print('Extracting information from dates....')
        print(self.Stock.shape)
        
    
    def get_IDXFunds_features(self, Ticker):
        print(' ')
        print('Fetching data for NASDAQ-100 Index Fund ETF QQQ & S&P 500 index ......')
        print(self.Stock.shape)
        # Nasdaq-100 Index Fund ETF QQQ
        #QQQ, QQQ_info = self.ts.get_daily('QQQ', outputsize='full') 
        #QQQ = QQQ.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'QQQ_Close', '5. volume': 'Volume' })
        #QQQ = QQQ.rename_axis(['Date'])
        Stock_obj = yf.Ticker('QQQ')
        QQQ = Stock_obj.history(start=self.Start_Date)
        QQQ = QQQ.rename(columns={'Close': 'QQQ_Close'})
        QQQ = QQQ.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
        #sorting index
        QQQ = QQQ.sort_index(ascending=True, axis=0)
        #slicing the data for 15 years from '2004-01-02' to today
        #QQQ = QQQ.loc[self.Start_Date:]
        QQQ['QQQ(t-1)'] = QQQ.QQQ_Close.shift(periods=1)
        QQQ['QQQ(t-2)'] =  QQQ.QQQ_Close.shift(periods=2)
        QQQ['QQQ(t-5)'] =  QQQ.QQQ_Close.shift(periods=5)

        QQQ['QQQ_MA10'] = QQQ.QQQ_Close.rolling(window=10).mean()
        #QQQ['QQQ_MA10_t'] = QQQ.QQQ_ClosePrev1.rolling(window=10).mean()
        QQQ['QQQ_MA20'] = QQQ.QQQ_Close.rolling(window=20).mean()
        QQQ['QQQ_MA50'] = QQQ.QQQ_Close.rolling(window=50).mean()
        print(QQQ.shape)
        


        #S&P 500 Index Fund 
        #SnP, SnP_info = self.ts.get_daily('INX', outputsize='full') 
        #SnP = SnP.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'SnP_Close', '5. volume': 'Volume' })
        #SnP = SnP.rename_axis(['Date'])
        #SnP = SnP.drop(columns=['Open', 'High', 'Low', 'Volume'])

        Stock_obj = yf.Ticker('^GSPC')
        SnP = Stock_obj.history(start=self.Start_Date)
        SnP = SnP.rename(columns={'Close': 'SnP_Close'})
        SnP = SnP.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
        
        #sorting index
        SnP = SnP.sort_index(ascending=True, axis=0)
        #slicing the data for 15 years from '2004-01-02' to today
        #SnP = SnP.loc[self.Start_Date:]
        SnP
        SnP['SnP(t-1))'] = SnP.SnP_Close.shift(periods=1)
        SnP['SnP(t-5)'] =  SnP.SnP_Close.shift(periods=5)
        print(SnP.shape)
        
        #S&P 500 Index Fund 
        #DJIA, DJIA_info = self.ts.get_daily('DJI', outputsize='full') 
        #DJIA = DJIA.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'DJIA_Close', '5. volume': 'Volume' })
        #DJIA = DJIA.rename_axis(['Date'])
        #DJIA = DJIA.drop(columns=['Open', 'High', 'Low', 'Volume'])

        Stock_obj = yf.Ticker('^DJI')
        DJIA = Stock_obj.history(start=self.Start_Date)
        DJIA = DJIA.rename(columns={'Close': 'DJIA_Close'})
        DJIA = DJIA.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
        
        
        #sorting index
        DJIA = DJIA.sort_index(ascending=True, axis=0)
        #slicing the data for 15 years from '2004-01-02' to today
        #DJIA = DJIA.loc[self.Start_Date:]
        DJIA
        DJIA['DJIA(t-1))'] = DJIA.DJIA_Close.shift(periods=1)
        DJIA['DJIA(t-5)'] =  DJIA.DJIA_Close.shift(periods=5)
        print(DJIA.shape)
        print(self.Stock.shape)
        
        #Merge index funds 
        IDXFunds = QQQ.merge(SnP, left_index=True, right_index=True)
        IDXFunds = IDXFunds.merge(DJIA, left_index=True, right_index=True)
        self.Stock = self.Stock.merge(IDXFunds, left_index=True, right_index=True)
        print(self.Stock.shape)
        
        
        
    def forcast_Horizon(self, Ticker):
    
        print(' ')
        print('Adding the future day close price as a target column for Forcast Horizon of ' + str(self.forcast_horz))
        #Adding the future day close price as a target column which needs to be predicted using Supervised Machine learning models
        self.Stock['Close_forcast'] = self.Stock.Close.shift(-self.forcast_horz)
        self.Stock = self.Stock.rename(columns={'Close': 'Close(t)'})
        self.Stock = self.Stock.dropna()
        print(self.Stock.shape)

        
    def save_features(self, Ticker):
        print('Saving extracted features data in S3 Bucket....')
        self.Stock.to_csv(self.Ticker + '.csv')
        print('Extracted features shape - '+ str(self.Stock.shape))
        print(' ')
        print('Extracted features dataframe - ')
        print(self.Stock)
        return self.Stock
        
        
    T = TypeVar('T')

    def pipeline(self,
        value: T,
        function_pipeline: Sequence[Callable[[T], T]],
        ) -> T:
    
        return reduce(lambda v, f: f(v), function_pipeline, value)

    def pipeline_sequence(self):

        print('Initiating Pipeline....')
        z = self.pipeline(
            value=self.Ticker,
            function_pipeline=(
                self.get_stock_data,
                self.extract_Technical_Indicators,
                self.extract_date_features, 
                self.get_IDXFunds_features,
                self.forcast_Horizon,
                self.save_features
                    )
                )

        print(f'z={z}')
        


class Stock_Prediction_Modeling():
    def __init__(self, Stocks, models, features):
        self.Stocks = Stocks
        self.train_Models = models
        self.metrics = {}
        self.features_selected = features
        
        
    def get_stock_data(self, Ticker):
        
        file = self.Ticker + '.csv'
        Stock = pd.read_csv(file,  index_col=0)
        print(Stock)
        #print(self.features_selected)
        print('Loading Historical Price data for ' + self.Ticker + '....')
        
        self.df_Stock = Stock.copy() #[features_selected]
        #self.df_Stock = self.df_Stock.drop(columns=['Date_col'])
        self.df_Stock = self.df_Stock[self.features_selected]
        
        self.df_Stock = self.df_Stock.rename(columns={'Close(t)':'Close'})
        
        #self.df_Stock = self.df_Stock.copy()
        self.df_Stock['Diff'] = self.df_Stock['Close'] - self.df_Stock['Open']
        self.df_Stock['High-low'] = self.df_Stock['High'] - self.df_Stock['Low']
        
        #print('aaaa')
        st.write('Training Selected Machine Learning models for ', self.Ticker)
        #features_selected = ['Close', 'Diff', 'High-low', 'QQQ_Close', 'SnP_Close','DJIA_Close', 'ATR', 'RSI', 'MA50', 'EMA200', 'Upper_Band']
        st.markdown('Your **_final_ _dataframe_ _for_ Training** ')
        st.write(self.df_Stock)
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        st.success('Training Completed!')

        #self.df_Stock = self.df_Stock[:-70]
        
        print(self.df_Stock.columns)


    def prepare_lagged_features(self, lag_stock, lag_index, lag_diff):

        print('Preparing Lagged Features for Stock, Index Funds.....')
        lags = range(1, lag_stock+1)
        lag_cols= ['Close']
        self.df_Stock=self.df_Stock.assign(**{
            '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
            for l in lags
            for col in lag_cols
        })

       
        lags = range(1, lag_index+1)
        lag_cols= ['QQQ_Close','SnP_Close','DJIA_Close']
        self.df_Stock= self.df_Stock.assign(**{
            '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
            for l in lags
            for col in lag_cols
        })

        self.df_Stock = self.df_Stock.drop(columns=lag_cols)


        lags = range(1, lag_diff+1)
        lag_cols= ['Diff','High-low']
        self.df_Stock= self.df_Stock.assign(**{
            '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
            for l in lags
            for col in lag_cols
        })

        self.df_Stock = self.df_Stock.drop(columns=lag_cols)

        remove_lags_na = max(lag_stock, lag_index, lag_diff) + 1
        print('Removing NAN rows - ', str(remove_lags_na))
        self.df_Stock = self.df_Stock.iloc[remove_lags_na:,]
        return self.df_Stock

    def get_lagged_features(self, Ticker):
        
        self.df_Stock_lagged = self.prepare_lagged_features(lag_stock = 20, lag_index = 10, lag_diff = 5)

        print(self.df_Stock_lagged.columns)
        
        self.df_Stock = self.df_Stock_lagged.copy()
        print(self.df_Stock.shape)
        print('Extracted Feature Columns after lagged effect - ')
        print(self.df_Stock.columns)
        
        '''
        self.df_Stock['Close'].plot(figsize=(10, 7))
        plt.title("Stock Price", fontsize=17)
        plt.ylabel('Price', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.show()
        print(self.df_Stock)
        '''



    def create_train_test_set(self):

        #self.df_Stock = self.df_Stock[:-60]
        self.features = self.df_Stock.drop(columns=['Close'], axis=1)
        self.target = self.df_Stock['Close']


        data_len = self.df_Stock.shape[0]
        print('Historical Stock Data length is - ', str(data_len))

        #create a chronological split for train and testing
        train_split = int(data_len * 0.9)
        print('Training Set length - ', str(train_split))

        val_split = train_split + int(data_len * 0.08)
        print('Validation Set length - ', str(int(data_len * 0.1)))

        print('Test Set length - ', str(int(data_len * 0.02)))

        # Splitting features and target into train, validation and test samples 
        X_train, X_val, X_test = self.features[:train_split], self.features[train_split:val_split], self.features[val_split:]
        Y_train, Y_val, Y_test = self.target[:train_split], self.target[train_split:val_split], self.target[val_split:]

        #print shape of samples
        print(X_train.shape, X_val.shape, X_test.shape)
        print(Y_train.shape, Y_val.shape, Y_test.shape)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def get_train_test(self):
        print('Splitting the data into Train and Test ...')
        print(' ')
        if self.ML_Model == 'LSTM':
            self.scale_LSTM_features()
            self.X_train, self.X_test, self.Y_train, self.Y_test = self.create_train_test_LSTM()
        else:
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.create_train_test_set()
            #print('here6')

    def get_mape(self, y_true, y_pred): 
        """
        Compute mean absolute percentage error (MAPE)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
    def calc_metrics(self):
        print('Evaluating Metrics - MAE, MAPE, RMSE, R Square')
        print(' ')
        if self.ML_Model == 'LSTM':
        
            self.Train_RSq = round(metrics.r2_score(self.Y_train,self.Y_train_pred),2)
            self.Train_EV = round(metrics.explained_variance_score(self.Y_train,self.Y_train_pred),2)
            self.Train_MAPE = round(self.get_mape(self.Y_train,self.Y_train_pred), 2)
            self.Train_MSE = round(metrics.mean_squared_error(self.Y_train,self.Y_train_pred), 2) 
            self.Train_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_train,self.Y_train_pred)),2)
            self.Train_MAE = round(metrics.mean_absolute_error(self.Y_train,self.Y_train_pred),2)

            
            self.Test_RSq = round(metrics.r2_score(self.Y_test,self.Y_test_pred),2)
            self.Test_EV = round(metrics.explained_variance_score(self.Y_test,self.Y_test_pred),2)
            self.Test_MAPE = round(self.get_mape(self.Y_test,self.Y_test_pred), 2)
            self.Test_MSE = round(metrics.mean_squared_error(self.Y_test,self.Y_test_pred), 2) 
            self.Test_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_test,self.Y_test_pred)),2)
            self.Test_MAE = round(metrics.mean_absolute_error(self.Y_test,self.Y_test_pred),2)
        else:
            #print('here6')
            self.Train_RSq = round(metrics.r2_score(self.Y_train,self.Y_train_pred),2)
            self.Train_EV = round(metrics.explained_variance_score(self.Y_train,self.Y_train_pred),2)
            self.Train_MAPE = round(self.get_mape(self.Y_train,self.Y_train_pred), 2)
            self.Train_MSE = round(metrics.mean_squared_error(self.Y_train,self.Y_train_pred), 2) 
            self.Train_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_train,self.Y_train_pred)),2)
            self.Train_MAE = round(metrics.mean_absolute_error(self.Y_train,self.Y_train_pred),2)

            self.Val_RSq = round(metrics.r2_score(self.Y_val,self.Y_val_pred),2)
            self.Val_EV = round(metrics.explained_variance_score(self.Y_val,self.Y_val_pred),2)
            self.Val_MAPE = round(self.get_mape(self.Y_val,self.Y_val_pred), 2)
            self.Val_MSE = round(metrics.mean_squared_error(self.Y_train,self.Y_train_pred), 2) 
            self.Val_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_val,self.Y_val_pred)),2)
            self.Val_MAE = round(metrics.mean_absolute_error(self.Y_val,self.Y_val_pred),2)

            self.Test_RSq = round(metrics.r2_score(self.Y_test,self.Y_test_pred),2)
            self.Test_EV = round(metrics.explained_variance_score(self.Y_test,self.Y_test_pred),2)
            self.Test_MAPE = round(self.get_mape(self.Y_test,self.Y_test_pred), 2)
            self.Test_MSE = round(metrics.mean_squared_error(self.Y_test,self.Y_test_pred), 2) 
            self.Test_RMSE = round(np.sqrt(metrics.mean_squared_error(self.Y_test,self.Y_test_pred)),2)
            self.Test_MAE = round(metrics.mean_absolute_error(self.Y_test,self.Y_test_pred),2)


    def update_metrics_tracker(self):
        print('Updating the metrics tracker....')
        if self.ML_Model == 'LSTM':
            #self.metrics[self.Ticker] = {}
            self.metrics[self.Ticker][self.ML_Model] = {'Train_MAE': self.Train_MAE, 'Train_MAPE': self.Train_MAPE , 'Train_RMSE': self.Train_RMSE,
                          'Test_MAE': self.Test_MAE, 'Test_MAPE': self.Test_MAPE, 'Test_RMSE': self.Test_RMSE}
        else:
            ##self.metrics[self.Ticker] = {{}}
            self.metrics[self.Ticker][self.ML_Model] = {'Train_MAE': self.Train_MAE, 'Train_MAPE': self.Train_MAPE , 'Train_RMSE': self.Train_RMSE,
                          'Test_MAE': self.Val_MAE, 'Test_MAPE': self.Val_MAPE, 'Test_RMSE': self.Val_RMSE}

       

    def train_model(self, Ticker):

        for model in self.train_Models:
            self.ML_Model = model
            if self.ML_Model == 'Linear Regression':
                
                print(' ')
                print('Training Linear Regressiom Model')
                
                self.get_train_test()
                
                
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(self.X_train, self.Y_train)
                print('LR Coefficients: \n', lr.coef_)
                print('LR Intercept: \n', lr.intercept_)

                print("Performance (R^2): ", lr.score(self.X_train, self.Y_train))

                self.Y_train_pred = lr.predict(self.X_train)
                self.Y_val_pred = lr.predict(self.X_val)
                self.Y_test_pred = lr.predict(self.X_test)

                self.calc_metrics()
                self.update_metrics_tracker()
                self.plot_prediction()
                
            elif self.ML_Model == 'XGBoost':
                print(' ')
                print('Training XGBoost Model')
                
                self.get_train_test()
                
                from xgboost import XGBRegressor
                n_estimators = 100             # Number of boosted trees to fit. default = 100
                max_depth = 10                 # Maximum tree depth for base learners. default = 3
                learning_rate = 0.2            # Boosting learning rate (xgb’s “eta”). default = 0.1
                min_child_weight = 1           # Minimum sum of instance weight(hessian) needed in a child. default = 1
                subsample = 1                  # Subsample ratio of the training instance. default = 1
                colsample_bytree = 1           # Subsample ratio of columns when constructing each tree. default = 1
                colsample_bylevel = 1          # Subsample ratio of columns for each split, in each level. default = 1
                gamma = 2                      # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0

                model_seed = 42



                xgb = XGBRegressor(seed=model_seed,
                                         n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         learning_rate=learning_rate,
                                         min_child_weight=min_child_weight,
                                         subsample=subsample,
                                         colsample_bytree=colsample_bytree,
                                         colsample_bylevel=colsample_bylevel,
                                         gamma=gamma)
                xgb.fit(self.X_train, self.Y_train)

                self.Y_train_pred = xgb.predict(self.X_train)
                self.Y_val_pred = xgb.predict(self.X_val)
                self.Y_test_pred = xgb.predict(self.X_test)

                self.calc_metrics()
                self.update_metrics_tracker()
                
                fig = plt.figure(figsize=(8,8))
                plt.xticks(rotation='vertical')
                plt.bar([i for i in range(len(xgb.feature_importances_))], xgb.feature_importances_.tolist(), tick_label=self.X_test.columns)
                plt.title('Feature importance of the technical indicators.')
                plt.show()
                
                self.plot_prediction()
                
            elif self.ML_Model == 'Random Forest':
                print(' ')
                print('Training Random Forest Model')
                
                self.get_train_test()
                rf = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42)
                rf.fit(self.X_train, self.Y_train)
                
                self.Y_train_pred = rf.predict(self.X_train)
                self.Y_val_pred = rf.predict(self.X_val)
                self.Y_test_pred = rf.predict(self.X_test)
                
                self.calc_metrics()
                self.update_metrics_tracker()
                self.plot_prediction()
                

                
                
    def plot_prediction(self):
        '''
        rcParams['figure.figsize'] = 10, 8 # width 10, height 8

        ax = self.Y_train.plot(x='Date', y='aa', style='b-', grid=True)
        ax = self.Y_val.plot(x='Date', y='adj_close', style='y-', grid=True, ax=ax)
        ax = self.Y_test.plot(x='Date', y='adj_close', style='g-', grid=True, ax=ax)
        ax.legend(['train', 'dev', 'test'])
        ax.set_xlabel("date")
        ax.set_ylabel("USD")
        '''
        print(' ')
        print('Predicted vs Actual for ', self.ML_Model)
        st.write('Predicted vs Actual for ', self.ML_Model)
        self.df_pred = pd.DataFrame(self.Y_val.values, columns=['Actual'], index=self.Y_val.index)
        self.df_pred['Predicted'] = self.Y_val_pred
        self.df_pred = self.df_pred.reset_index()
        self.df_pred.loc[:, 'Date'] = pd.to_datetime(self.df_pred['Date'],format='%Y-%m-%d')
        print('Stock Prediction on Test Data - ',self.df_pred)
        st.write('Stock Prediction on Test Data for - ',self.Ticker)
        st.write(self.df_pred)

        print('Plotting Actual vs Predicted for - ', self.ML_Model)
        st.write('Plotting Actual vs Predicted for - ', self.ML_Model)
        fig = self.df_pred[['Actual', 'Predicted']].plot()
        plt.title('Actual vs Predicted Stock Prices')
        #plt.show()
        #st.write(fig)
        st.pyplot()


    
    def save_results(self, Ticker):
        import json
        print('Saving Metrics in Json for Stock - ', self.Ticker)
        with open('./metrics.txt', 'w') as json_file:
            json.dump(self.metrics, json_file)
        
    
    def pipeline(self,
        value: T,
        function_pipeline: Sequence[Callable[[T], T]],
        ) -> T:
    
        return reduce(lambda v, f: f(v), function_pipeline, value)

    def pipeline_sequence(self):
        for stock in self.Stocks:
            self.Ticker = stock
            self.metrics[self.Ticker] = {}
            print('Initiating Pipeline for Stock Ticker ---- ', self.Ticker)
            st.write('Initiating Pipeline for Stock Ticker ---', self.Ticker)
            z = self.pipeline(
                value=self.Ticker,
                function_pipeline=(
                    self.get_stock_data,
                    self.get_lagged_features,
                    self.train_model, 
                    self.save_results
                        )
                    )

            print(f'z={z}')

def stock_financials(stock):
    df_ticker = yf.Ticker(stock)
    sector = df_ticker.info['sector']
    prevClose = df_ticker.info['previousClose']
    marketCap = df_ticker.info['marketCap']
    twoHunDayAvg = df_ticker.info['twoHundredDayAverage']
    fiftyTwoWeekHigh = df_ticker.info['fiftyTwoWeekHigh']
    fiftyTwoWeekLow = df_ticker.info['fiftyTwoWeekLow']
    Name = df_ticker.info['longName']
    averageVolume = df_ticker.info['averageVolume']
    shortRatio = df_ticker.info['shortRatio']
    ftWeekChange = df_ticker.info['52WeekChange']
    website = df_ticker.info['website']
    
    
    st.write('Company Name -', Name)
    st.write('Sector -', sector)
    st.write('Company Website -', website)
    st.write('Average Volume -', averageVolume)
    st.write('Market Cap -', marketCap)
    st.write('Previous Close -', prevClose)
    st.write('52 Week Change -', ftWeekChange)
    st.write('52 Week High -', fiftyTwoWeekHigh)
    st.write('52 Week Low -', fiftyTwoWeekLow)
    st.write('200 Day Average -', twoHunDayAvg)
    st.write('Short Ratio -', shortRatio)
    
    
def plot_time_series(stock):
    df_ticker = yf.Ticker(stock)
    data = df_ticker.history()
    data = data.sort_index(ascending=True, axis=0)
    data['Date'] = data.index
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open",line_color='crimson'))
    fig.add_trace(go.Scatter(x=data.Date, y=data['Close'], name="stock_close",line_color='dimgray'))
    fig.add_trace(go.Scatter(x=data.Date, y=data['High'], name="stock_high",line_color='blueviolet'))
    fig.add_trace(go.Scatter(x=data.Date, y=data['Low'], name="stock_low",line_color='darksalmon'))

    fig.layout.update(title_text='Stock Price with Rangeslider',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

df_stock = pd.DataFrame()
eval_metrics = {}

st.title("Stock Prediction - More Money than God!")

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #0A3648;

}
</style>
    """, unsafe_allow_html=True)
#0A3648
#13393E
menu=["Stocks Exploration & Feature Extraction", "Train Your Own Drogon (Machine Learning Models)","Look Into The Far Future (LSTM)"]
choices = st.sidebar.selectbox("Select Dashboard",menu)



if choices == 'Stocks Exploration & Feature Extraction':
    st.subheader('Stock Exploration & Feature extraction')
    st.sidebar.success("Greed, for lack of a better word, is good")

    st.write('Feature Extraction is a tedious job to do more so when we are talking about stocks. We have \
                 created this Pipeline to extract many Technical Indicators as well as create lagged features \
                 for training a Machine Learning algorithm for forcasting Stock Prices.')
    user_input = ''
    st.markdown('Enter **_Ticker_ Symbol** for the **Stock**')
    user_input = st.text_input("", '')
    
    if not user_input:
            pass
    else:
        
        st.markdown('Select from the options below to Explore Stocks')
        
        selected_explore = st.selectbox("", options=['Select your Option', 'Stock Financials Exploration', 'Extract Features for Stock Price Forecasting'], index=0)
        if selected_explore == 'Stock Financials Exploration':
            st.markdown('')
            st.markdown('**_Stock_ Financial** Information')
            st.markdown('')
            st.markdown('')
            stock_financials(user_input)
            plot_time_series(user_input)

    
        elif selected_explore == 'Extract Features for Stock Price Forecasting':
            

            st.markdown('**_Real-Time_ _Feature_ Extraction** for any Stocks')
            
            st.write('Select a Date from a minimum of a year before as some of the features we extract uses upto 200 days of data. ')
            st.markdown('Select **_Start_ _Date_ _for_ _Historical_ Stock** Data & features')
            start_date = st.date_input(
            "", datetime(2015, 5, 4))
            st.write('You selected data from -', start_date)

            submit = st.button('Extract Features')
            if submit:
                try:
                    
                    with st.spinner('Extracting Features... '):
                        time.sleep(2)
                    print('Date - ', start_date)
                    features = Stocks(user_input, start_date, 1)
                    features.pipeline_sequence()

                except:
                    st.markdown('If you wants to make money, your **_Ticker_ symbol** should be correct!!! :p ')
                file_name = user_input + '.csv'
                df_stock = pd.read_csv(file_name)
                st.write('Extracted Features Dataframe for ', user_input)
                st.write(df_stock)
                #st.write('Download Link')

                st.write('We have extracted', len(df_stock.columns), 'columns for this stock. You can Analyse it or even train it for Stock Prediction.')


                st.write('Extracted Feature Columns are', df_stock.columns)

elif choices == 'Train Your Own Drogon (Machine Learning Models)':
    st.subheader('Train Machine Learning Models for Stock Prediction & Generate your own Buy/Sell Signals using the best Model')
    st.sidebar.success("The most valuable commodity I know of is information.")

    
    st.markdown('**_Real_ _Time_ ML Training** for any Stocks')
    st.write('We have created this pipeline for multiple Model training on Multiple stocks at the same time and evaluating them')

    
    st.write('Make sure you have Extracted features for the Stocks you want to train models on using first Tab')
    
    result = glob.glob( '*.csv' )
    #st.write( result )
    stock = []
    for val in result:
        stock.append(val.split('.')[0])
    
    st.markdown('**_Recently_ _Extracted_ Stocks** -')
    st.write(stock[:5])
    cols1 = ['NKE', 'JNJ']
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    Stocks = st.multiselect("", stock, default=cols1)
    
    options = ['Linear Regression', 'Random Forest', 'XGBoost']
    cols2 = ['Linear Regression', 'Random Forest']
    st.markdown('**_Select_ _Machine_ _Learning_ Algorithms** to Train')
    models = st.multiselect("", options, default=cols2)
    
    
    file = './' + stock[0] + '.csv'
    df_stock = pd.read_csv(file)
    df_stock = df_stock.drop(columns=['Date', 'Date_col'])
    #st.write(df_stock.columns)
    st.markdown('Select from your **_Extracted_ features** or use default')
    st.write('Select all Extracted features')
    all_features = st.checkbox('Select all Extracted features')
    cols = ['Open', 'High', 'Low', 'Close(t)', 'Upper_Band', 'MA200', 'ATR', 'ROC', 'QQQ_Close', 'SnP_Close', 'DJIA_Close', 'DJIA(t-5)']
    if all_features:
        cols = df_stock.columns.tolist()
        cols.pop(len(df_stock.columns)-1)

    features = st.multiselect("", df_stock.columns.tolist(), default=cols)
    
    
    submit = st.button('Train Your DROGON')
    if submit:
        try:
            training = Stock_Prediction_Modeling(Stocks, models, features)
            training.pipeline_sequence()
            with open('./metrics.txt') as f:
                eval_metrics = json.load(f)

            

        except:
            st.markdown('There seems to be a error - **_check_ logs**')
            print("Unexpected error:", sys.exc_info())
            print()

    
        Metrics = pd.DataFrame.from_dict({(i,j): eval_metrics[i][j] 
                               for i in eval_metrics.keys() 
                               for j in eval_metrics[i].keys()},
                           orient='index')

        st.write(Metrics)
    
    
    
    
elif choices == 'Look Into The Far Future (LSTM)':
    st.subheader('Look Into The Future to Predict Stock Prices for Any Stocks and Generate Buy/Sell Signals')
    st.sidebar.success("This is a Work In Progress, will be added shortly")
