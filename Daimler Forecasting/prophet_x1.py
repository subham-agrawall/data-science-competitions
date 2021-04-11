import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn import metrics

data = pd.read_excel('TrainingSet_IITM.xlsx', parse_dates=[0], index_col=0)
test = pd.read_excel('TestSet_IITM.xlsx', parse_dates=[0], index_col=0)

init='2016-07-01'
data=data.append(test)
dates=list(pd.date_range(init,'2017-07-31'))
clean_data=pd.DataFrame(index=dates)
clean_data['X1']=data['X1']
clean_data['X2']=data['X2']
data=clean_data

df=pd.DataFrame()
df['ds']=data.index
df['y']=np.log(data['X1'].values)
df.head()

m = Prophet()
m.add_seasonality(name='yearly', period=368, fourier_order=8)
m.fit(df)
future = m.make_future_dataframe(periods=61)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)
m.plot_components(forecast)
plt.show()

pred1=forecast[['ds', 'yhat']][-61:]
pred1.index=pred1['ds']
del pred1['ds']
pred1['pred']=np.exp(pred1['yhat'])
del pred1['yhat']
dat=list(pd.bdate_range('2017-08-01','2017-09-30'))
pred1=pred1[pred1.index.isin(dat)]

dats=list(pd.date_range('2017-08-01','2017-09-30'))
prediction=pd.DataFrame(index=dats)
prediction['X1']=pred1['pred']