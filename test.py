from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM


import datetime as dt
import pandas as pd
import pandas_datareader as web
import numpy as np
company = 'FB'


start = dt.datetime(2012,1,1)
end = dt.datetime(2019.12,31)

data = web.DataReader(company, 'google', start, end)

scalar = MinMaxScaler(feature_range=(0,1))
scale_data = scalar.fit_transform(data['Close'].values.reshape(-1, 1))


prediction_data = 60


x_train = []
y_train = []


for x in range(prediction_data, len(scale_data)):
    x_train.append(scale_data[x-prediction_data:x])
    y_train.append(scale_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

