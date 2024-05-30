
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



