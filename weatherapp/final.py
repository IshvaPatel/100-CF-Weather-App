import requests
import pandas as pd
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt
import datetime


class Forecasting():

    def __init__(self,city):
        self.city=city
        self.lat=None
        self.lon=None
        self.data=None
        self.model=None
        self.scalar=None
        self.X_train=None
        self.X_test =None
        self.y_train=None
        self.y_test=None
        self.seq_length=None
    @staticmethod

    def getHistoricalHourlyData(self):
        bearer_token = os.environ.get('7dfcf22c94fa8abff845e481a769bef4')
        headers = {"Authorization": "Bearer {}".format(bearer_token)}
        geoencodingUrl = f"http://api.openweathermap.org/geo/1.0/direct?q={self.city}&limit=1&appid=7dfcf22c94fa8abff845e481a769bef4"
        landmarks=requests.request("GET",geoencodingUrl,headers=headers).json()
        self.lat , self.lon={landmarks[0]['lat'],landmarks[0]['lon']}
        url = f"http://history.openweathermap.org/data/2.5/history/city?lat={self.lat}&lon={self.lon}&type=hour&appid=7dfcf22c94fa8abff845e481a769bef4"
        response = requests.request("GET", url, headers=headers).json()
        df=pd.DataFrame(columns=['Date','Temperature','Humidity','Wind Speed'])
        for val in response['list']:
            date_str = datetime.datetime.fromtimestamp(val['dt']).strftime("%H:%M:%S %d/%B/%Y")
            df = df.append({'Date': date_str, 'Temperature': val['main']['temp'], 'Humidity': val['main']['humidity'],'Wind Speed':val['wind']['speed']}, ignore_index=True)

        df['Date']=pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        self.data=df
    
    def preprocessing(self):
        df=self.data
        self.scaler = MinMaxScaler()
        df[['Temperature', 'Humidity', 'Wind Speed']] = self.scaler.fit_transform(df[['Temperature', 'Humidity', 'Wind Speed']])
        def create_sequences(df, seq_length):
            X, y = [], []
            for i in range(len(df) - seq_length):
                X.append(df[i:i + seq_length])
                y.append(df[i + seq_length])
            return np.array(X), np.array(y)

        self.seq_length = 10  # Example sequence length
        X, y = create_sequences(df.values, self.seq_length)

        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]

        self.y_train, self.y_test = y[:train_size], y[train_size:]

    
    def LSTM(self):
        self.preprocessing(self)
        self.model = Sequential()
        self.model.add(LSTM(units=50,activation='relu', return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.y_train.shape[1])) 
        self.model.compile(optimizer='adam', loss='mse')
        #model.summary()
        history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, validation_split=0.2)
        loss = self.model.evaluate(self.X_test, self.y_test)

    def plot(self):
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        # Inverse transform the predictions
        train_predictions = self.scaler.inverse_transform(train_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        data=self.scaler.inverse_transform(data)
        train_predictions=pd.DataFrame(train_predictions)
        test_predictions=pd.DataFrame(test_predictions)

        plt.figure(figsize=(10, 6))

        plt.plot(self.data.index[self.seq_length:], self.data[0][self.seq_length:], label='Actual', color='blue')

        # Plot training predictions
        plt.plot(self.data.index[self.seq_length:self.seq_length+len(train_predictions[0])], train_predictions[0], label='Train Predictions',color='green')

        # Plot testing predictions
        test_pred_index = range(self.seq_length+len(train_predictions[0]), self.seq_length+len(train_predictions[0])+len(test_predictions[0]))
        plt.plot(self.data.index[test_pred_index], test_predictions[0], label='Test Predictions',color='orange')
        plt.legend()
        plt.show()

    def generate_forecast(self, forecast_period=10):
        forecast = []

        # Use the last sequence from the test data to make predictions
        last_sequence = self.X_test[-1]
        
        for _ in range(forecast_period):
            # Reshape the sequence to match the input shape of the model
            current_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
            
            # Predict the next value
            next_prediction = self.model.predict(current_sequence)[0]
            
            # Append the prediction to the forecast list
            forecast.append(next_prediction)
            
            # Update the last sequence by removing the first element and appending the predicted value
            last_sequence = np.append(last_sequence[1:], [next_prediction], axis=0)

        # Inverse transform the forecasted values
        original_forecasts = self.scaler.inverse_transform(forecast)

        # Create a DataFrame for better readability
        forecast_dates = pd.date_range(start=self.data.index[-1], periods=forecast_period + 1, closed='right')
        forecast_df = pd.DataFrame(original_forecasts, columns=['Temperature', 'Humidity', 'Wind Speed'], index=forecast_dates)

        return forecast_df

