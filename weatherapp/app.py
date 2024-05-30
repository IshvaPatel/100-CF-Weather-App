from flask import Flask, request, jsonify, render_template,redirect
import os
from final1 import Forecasting  # Ensure this imports the Forecasting class correctly

app = Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        city = request.form.get('city')
        if city:
            forecasting = Forecasting(city)
            forecasting.getHistoricalHourlyData()
            forecasting.LSTM()
            forecast_df = forecasting.generate_forecast(forecast_period=10)

            # Convert index to string for template
            forecast_df.index = forecast_df.index.strftime('%Y-%m-%d %H:%M:%S')
            forecast_data = forecast_df.reset_index().to_dict(orient='records')

            return render_template('index.html', city=city, forecast_data=forecast_data)
        else:
            return "City not provided", 400
    else:
        # Handle GET request, maybe redirect to home page
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
