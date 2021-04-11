# Forecasting
This repo contains codes and report created for time series competition by Daimler.  

## Problem Statement
Analysis and forecasting of two different time series with daily frequency.  

## Techniques
1. Visualization and analysis of time series - data_visualization.py    
    This includes yearly plots, monthly plots, lag scatter plots, autocorrelation plots and stationrity checks.  
2. Baseline forecast - baseline.py      
    MSE error was calculated by forecasting with lag 1.  
3. SARIMA - seasonal.py  
    Implementation of Seasonal ARIMA in python.  
4. AR - autoregression.py  
    Implementation of Autoregression (AR) models in python.  
    Link for reference - https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/  
5. Prophet - prophet.py  
    Implementation of prophet in python. This performed best in this case for both time series.  
    Transformation and parameters are different for both time series.  
    
## Report
Final report - report.pdf  
This report includes, detailed approach and final forecasts with reasonings.
