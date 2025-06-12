import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# Load the data
df = pd.read_csv("data/Historical Data_Site1.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Total Solar Generation': 'y'})

# Create and fit the model
model = Prophet(yearly_seasonality=True, daily_seasonality=True, weekly_seasonality=True)
model.fit(df)

# Create future dataframe for 2024
future = model.make_future_dataframe(periods=366, freq='D')  # 366 days for leap year 2024
forecast = model.predict(future)

# Filter for 2024
forecast_2024 = forecast[forecast['ds'].dt.year == 2024]

# Daily prediction for 2024
daily_prediction = forecast_2024[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
daily_prediction = daily_prediction.rename(columns={'ds': 'Date', 'yhat': 'Predicted_kWh', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'})

# Monthly prediction for 2024
monthly_prediction = forecast_2024.set_index('ds').resample('M')['yhat'].sum().reset_index()
monthly_prediction.columns = ['Month', 'Predicted_kWh']
monthly_prediction['Month'] = monthly_prediction['Month'].dt.strftime('%B %Y')

# Plotting
fig = go.Figure()

fig.add_trace(go.Scatter(x=daily_prediction['Date'], y=daily_prediction['Predicted_kWh'],
                         mode='lines', name='Daily Prediction'))

fig.add_trace(go.Scatter(x=daily_prediction['Date'], y=daily_prediction['Upper_Bound'],
                         mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=daily_prediction['Date'], y=daily_prediction['Lower_Bound'],
                         mode='lines', line=dict(width=0), 
                         fill='tonexty', fillcolor='rgba(0,100,80,0.2)', name='Confidence Interval'))

fig.update_layout(title='Daily Solar Energy Production Forecast for 2024',
                  xaxis_title='Date', yaxis_title='Predicted Solar Energy (kWh)',
                  legend_title='Legend')

fig.show()

print("Monthly Solar Energy Production Forecast for 2024:")
print(monthly_prediction.to_string(index=False))

# Save predictions to CSV
daily_prediction.to_csv('daily_solar_prediction_2024.csv', index=False)
monthly_prediction.to_csv('monthly_solar_prediction_2024.csv', index=False)