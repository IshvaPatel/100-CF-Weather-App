
# Weather Forecast Web Application

## Overview

This web application retrieves current weather information from OpenWeatherMap and provides users with the option to generate a forecast for the next ten hours using a Long Short-Term Memory (LSTM) model trained on historical hourly data from OpenWeatherMap.

## Features

- Retrieve current weather information from OpenWeatherMap API.
- Generate a forecast for the next ten hours using an LSTM model trained on historical data.
- User-friendly interface for entering city names and viewing weather forecasts.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/weather-forecast-app.git
    ```

2. Navigate to the project directory:

    ```bash
    cd weather-forecast-app
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the web server:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://localhost:5000` to access the web application.

3. Enter a city name in the input field and click on the "Get Weather" button to retrieve current weather information.

4. Optionally, click on the "Generate Forecast" button to generate a forecast for the next ten hours based on historical data.

## Improvements
Increase Dataset Size: Retrieving a larger dataset can provide more diverse examples for training, leading to better model performance.

Add More LSTM Layers: Increasing the depth of the neural network architecture by adding more LSTM layers can help capture more complex patterns in the data, improving the model's predictive capabilities.

Pretrained Models: Utilizing pretrained models trained on larger datasets can offer better performance out-of-the-box. Fine-tuning or transfer learning techniques can be applied to adapt the pretrained model to the specific dataset if necessary.

## Future Works

Integration with Multiple Data Sources: Extend the model to retrieve data from multiple sources to enrich the dataset and improve forecasting accuracy.

Model Evaluation and Hyperparameter Tuning: Implement thorough model evaluation techniques and hyperparameter tuning to optimize the model's performance.

Deployment: Develop methods for deploying the model in production environments, such as containerization or serverless architectures.

Real-time Forecasting: Implement real-time forecasting capabilities to provide up-to-date predictions as new data becomes available.





