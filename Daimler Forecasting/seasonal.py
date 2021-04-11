import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_excel('TrainingSet_IITM.xlsx', parse_dates=[0], index_col=0)
print(data.describe())

dates=pd.bdate_range('2014-01-02','2017-05-31')
clean_data=pd.DataFrame(index=dates)
clean_data['X1']=data['X1']
clean_data['X2']=data['X2']
data=clean_data
data=data.fillna(0)

# plt.figure()
# a=np.diff(data['X2'])
# date=[i.date() for i in data.index[1:]]
# plt.plot(date,a)
# plt.axhline(y=0, color='r', linestyle='-')
# plt.show()
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# series=pd.Series(np.diff(data['X1']))
# plot_acf(series,lags=400)
# plot_pacf(series,lags=400)
# plt.show()

mod = sm.tsa.statespace.SARIMAX(data['X2'], order=(1,1,0), seasonal_order=(1,1,0,366))
results = mod.fit()
print results.summary()

data['forecast'] = results.predict(start = 700, end= 890, dynamic= True)  
data[['logX1', 'forecast']].plot(figsize=(12, 8))
plt.show()

future_dates=pd.bdate_range('2017-06-01','2017-07-31')
future = pd.DataFrame(index=future_dates,columns= data.columns)
data = pd.concat([data, future])
data['forecast'] = results.predict(start = 890, end = 933, dynamic= True)
data[['logX1', 'forecast']].plot(figsize=(12, 8))
plt.show()

test = pd.read_excel('TestSet_IITM.xlsx', parse_dates=[0], index_col=0)
print(test.describe())
test['forecast']=data['forecast']
test['pred_X1']=np.exp(test['forecast'])
test[['X1', 'pred_X1']].plot(figsize=(12, 8))
plt.show()
from sklearn import metrics
print(metrics.mean_squared_error(test['X1'],test['pred_X1']))