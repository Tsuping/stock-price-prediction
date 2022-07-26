from cProfile import label
from pyexpat.errors import XML_ERROR_FEATURE_REQUIRES_XML_DTD
from matplotlib import units
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM


import datetime as dt
import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
company = 'FB'


start = dt.datetime(2011,1,1)
end = dt.datetime(2021,12,31)

data = web.DataReader(company, 'yahoo', start, end)

scalar = MinMaxScaler(feature_range=(0,1))
scale_data = scalar.fit_transform(data['Close'].values.reshape(-1, 1))


prediction_data = 60


x_train = []
y_train = []


for x in range(prediction_data, len(scale_data)):
    x_train.append(scale_data[x-prediction_data:x])
    y_train.append(scale_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=27, batch_size=45)

test_start = dt.datetime(2021,12,31)
test_end = dt.datetime.now()


test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_price = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_data:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scalar.transform(model_inputs)


x_test = []
for x in range(prediction_data, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_data:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predicted_price = model.predict(x_test)
predicted_price = scalar.inverse_transform(predicted_price)


plt.plot(actual_price, color="black", label=f"Actual {company} price")
plt.plot(predicted_price, color="green", label=f"Predicted {company} price")

plt.legend()
plt.show()