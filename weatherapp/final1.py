import requests
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import datetime


class Forecasting:
    def __init__(self, city):
        self.city = city
        self.lat = None
        self.lon = None
        self.data = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.seq_length = 10  # Example sequence length

    def getHistoricalHourlyData(self):
        api_key = "7dfcf22c94fa8abff845e481a769bef4"
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={self.city}&limit=1&appid={api_key}"
        response = requests.get(geo_url).json()
        self.lat, self.lon = response[0]['lat'], response[0]['lon']
        url = f"http://history.openweathermap.org/data/2.5/history/city?lat={self.lat}&lon={self.lon}&type=hour&appid={api_key}"
        response = requests.get(url).json()
        
        data_frames = []

        # Loop through each value in the response and create a DataFrame for each
        for val in response['list']:
            date_str = datetime.datetime.fromtimestamp(val['dt']).strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([{
                'Date': date_str,
                'Temperature': val['main']['temp'],
                'Humidity': val['main']['humidity'],
                'Wind Speed': val['wind']['speed']
            }])
            data_frames.append(df)

        # Concatenate all DataFrames in the list into a single DataFrame
        final_df = pd.concat(data_frames, ignore_index=True)
        final_df['Date'] = pd.to_datetime(final_df['Date'])
        final_df.set_index('Date', inplace=True)
        
        # Filter out rows where the index is NaT
        self.data = final_df[final_df.index.notna()]

    def preprocessing(self):
        df = self.data.copy()
        self.scaler = MinMaxScaler()
        df[['Temperature', 'Humidity', 'Wind Speed']] = self.scaler.fit_transform(df[['Temperature', 'Humidity', 'Wind Speed']])
        
        def create_sequences(df, seq_length):
            X, y = [], []
            for i in range(len(df) - seq_length):
                X.append(df[i:i + seq_length])
                y.append(df[i + seq_length])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(df.values, self.seq_length)
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

    def LSTM(self):
        self.preprocessing()
        self.model = Sequential()
        self.model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.y_train.shape[1])) 
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_split=0.2)

    def generate_forecast(self, forecast_period=10):
        forecast = []
        last_sequence = self.X_test[-1]
        for _ in range(forecast_period):
            current_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
            next_prediction = self.model.predict(current_sequence)[0]
            forecast.append(next_prediction)
            last_sequence = np.append(last_sequence[1:], [next_prediction], axis=0)
        original_forecasts = self.scaler.inverse_transform(forecast)
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        # Ensure the last index is not NaT
        if pd.isna(self.data.index[-1]):
            raise ValueError("The last date in the data index is NaT. Please provide a valid datetime index.")

        # Generate the forecast dates
        forecast_dates = pd.date_range(start=self.data.index[-1], periods=forecast_period+1, inclusive='right')
        forecast_df = pd.DataFrame(original_forecasts, columns=['Temperature', 'Humidity', 'Wind Speed'], index=forecast_dates)
        return forecast_df

if __name__ == "__main__":
    forecasting = Forecasting('New York')
    forecasting.getHistoricalHourlyData()
    forecasting.LSTM()
    forecast_df = forecasting.generate_forecast(forecast_period=10)
    print(forecast_df)
