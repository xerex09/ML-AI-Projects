import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import seaborn as sns
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


    

def load_and_preprocess_data(file_name):
    df = pd.read_csv(file_name, parse_dates=['Date'], index_col='Date')
    df.dropna(inplace=True)
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    return df

def plot_total_generation(df):
    plt.figure(figsize=(12, 6))
    df['Total Solar Generation'].plot()
    plt.title("Total Solar Generation vs Date")
    plt.xlabel("Date")
    plt.ylabel("Total Solar Generation")
    plt.grid(True)
    plt.show()

def plot_monthly_generation(df):
    monthly_generation = df.groupby('Month')['Total Solar Generation'].sum()
    plt.figure(figsize=(12, 6))
    monthly_generation.plot(kind='bar', color='skyblue')
    plt.title('Total Solar Generation by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Solar Generation (kWh)')
    plt.show()

def plot_yearly_generation(df):
    plt.figure(figsize=(12, 6))
    for year in df['Year'].unique():
        df_year = df[df['Year'] == year]
        df_grouped = df_year.groupby('Month')['Total Solar Generation'].sum()  
        plt.plot(df_grouped.index, df_grouped.values, label=str(year))
    plt.title("Total Solar Generation vs Month for Each Year")
    plt.xlabel("Month")
    plt.ylabel("Total Solar Generation")
    plt.legend(title="Year")
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.grid(True)
    plt.show()

def plot_correlation_analysis(df):
    correlation_matrix = df[['Total Solar Generation', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 
                             'temperature_2m_mean (°C)', 'sunshine_duration (s)', 'rain_sum (mm)']].corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    plt.figure(figsize=(14, 10))
    features = ['sunshine_duration (s)', 'temperature_2m_max (°C)', 'rain_sum (mm)']
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(data=df, x=feature, y='Total Solar Generation')
        plt.title(f'Total Solar Generation vs. {feature}')
    plt.tight_layout()
    plt.show()

def perform_dickey_fuller_test(df):
    result = adfuller(df['Total Solar Generation'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

def linear_regression(df):
    features = ['sunshine_duration (s)', 'temperature_2m_max (°C)', 'temperature_2m_mean (°C)']
    X = df[features]
    y = df['Total Solar Generation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Linear Regression MAE: {mae}')
    print(f'Linear Regression RMSE: {rmse}')

    plt.figure(figsize=(12, 6))
    plt.plot(X_test.index, y_test, label='Actual Total Solar Generation', color='blue')
    plt.plot(X_test.index, y_pred, label='Predicted Total Solar Generation (Linear Regression)', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Total Solar Generation')
    plt.title('Linear Regression Predictions vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

def sarima_model(df):
    split_date = df.index[-30]
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    sarima_model = SARIMAX(train_df['Total Solar Generation'], 
                           order=(1, 1, 1), 
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit()

    forecast = sarima_model.get_forecast(steps=len(test_df))
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    mae = mean_absolute_error(test_df['Total Solar Generation'], forecast_mean)
    rmse = np.sqrt(mean_squared_error(test_df['Total Solar Generation'], forecast_mean))
    print(f'SARIMA MAE: {mae}')
    print(f'SARIMA RMSE: {rmse}')

    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Total Solar Generation'], label='Actual Total Solar Generation', color='blue')
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast (SARIMA)', color='red', linestyle='dashed')
    plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='red', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Total Solar Generation')
    plt.title('SARIMA Forecast vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

def prophet_model(df):
    df_prophet = df[['Total Solar Generation']].reset_index().rename(columns={'Date': 'ds', 'Total Solar Generation': 'y'})
    split_date = df_prophet['ds'].max() - pd.DateOffset(months=1)
    train_prophet = df_prophet[df_prophet['ds'] <= split_date]
    test_prophet = df_prophet[df_prophet['ds'] > split_date]

    prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(train_prophet)

    future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='D')
    forecast = prophet_model.predict(future)

    forecast_test = forecast[['ds', 'yhat']].set_index('ds').join(test_prophet.set_index('ds'), how='inner')
    mae = mean_absolute_error(forecast_test['y'], forecast_test['yhat'])
    rmse = np.sqrt(mean_squared_error(forecast_test['y'], forecast_test['yhat']))
    print(f'Prophet MAE: {mae}')
    print(f'Prophet RMSE: {rmse}')

    fig = prophet_model.plot(forecast)
    plt.title('Prophet Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Total Solar Generation')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(test_prophet['ds'], test_prophet['y'], label='Actual Total Solar Generation', color='blue')
    plt.plot(forecast_test.index, forecast_test['yhat'], label='Forecasted Total Solar Generation', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Total Solar Generation')
    plt.title('Prophet Model Test Data vs Forecasted Data')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file_name = "data/Historical Data_Site1.csv"
    df = load_and_preprocess_data(file_name)
    
    plot_total_generation(df)
    plot_monthly_generation(df)
    plot_yearly_generation(df)
    
    plot_acf(df['Total Solar Generation'])
    plt.title('Autocorrelation of Total Solar Generation')
    plt.show()
    
    plot_pacf(df['Total Solar Generation'])
    plt.title('Partial Autocorrelation of Total Solar Generation')
    plt.show()
    
    result = seasonal_decompose(df['Total Solar Generation'], model='additive', period=30)
    result.plot()
    plt.show()
    
    plot_correlation_analysis(df)
    perform_dickey_fuller_test(df)
    linear_regression(df)
    sarima_model(df)
    
    auto_arima_model = pm.auto_arima(df['Total Solar Generation'], seasonal=True, m=12, stepwise=True, trace=True)
    print(auto_arima_model.summary())
    
    prophet_model(df)

if __name__ == "__main__":
    main()