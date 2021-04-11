import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn import metrics

data = pd.read_excel('TrainingSet_IITM.xlsx', parse_dates=[0], index_col=0)
test = pd.read_excel('TestSet_IITM.xlsx', parse_dates=[0], index_col=0)

init='2014-01-02'
data=data.append(test)
dates=list(pd.date_range(init,'2017-07-31'))
clean_data=pd.DataFrame(index=dates)
clean_data['X1']=data['X1']
clean_data['X2']=data['X2']
data=clean_data

df=pd.DataFrame()
df['ds']=data.index
df['y']=data['X2'].values
df.head()

m = Prophet(changepoint_prior_scale=0.05)
m.add_seasonality(name='yearly', period=366, fourier_order=38)
m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
m.add_seasonality(name='new', period=429, fourier_order=38,prior_scale=0.1)
m.fit(df)
future = m.make_future_dataframe(periods=61)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)
m.plot_components(forecast)
plt.show()

pred2=forecast[['ds', 'yhat']][-61:]
pred2.index=pred2['ds']
del pred2['ds']
pred2['pred']=pred2['yhat']
del pred2['yhat']
dat=list(pd.bdate_range('2017-08-01','2017-09-30'))
pred2=pred2[pred2.index.isin(dat)]
