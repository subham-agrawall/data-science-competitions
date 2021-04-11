import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

data = pd.read_excel('TrainingSet_IITM.xlsx', parse_dates=[0], index_col=0)
print(data.describe())

dates=pd.date_range('2014-01-02','2017-05-31')
clean_data=pd.DataFrame(index=dates)
clean_data['X1']=data['X1']
clean_data['X2']=data['X2']
data=clean_data
# data=data.fillna(1)

plt.figure()
a=np.diff(np.log(data['X1']))
date=[i.date() for i in data.index[1:]]
plt.plot(date,a)
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

plt.figure()
plt.plot_date(dates,data['X2']) # An outlier
plt.figure()
plt.scatter(data['X1'],data['X2']) # No relation
plt.show()

## Histograms
plt.figure()
data['X1'].hist()
plt.figure()
data['X2'].hist()
## Density plots
plt.figure()
data['X1'].plot(kind='kde')
plt.figure()
data['X2'].plot(kind='kde') #Approximately Normal-Check (remove outlier)
plt.show()

## YEARLY PLOTS##################
dates=pd.date_range('2014-01-01','2016-12-31')
clean_data=pd.DataFrame(index=dates)
clean_data['X1']=np.log(data['X1'])
clean_data['X2']=data['X2']
data=clean_data
data=data.fillna(0)
data=data.drop(pd.Timestamp('2016-02-29 00:00:00'))

ind=data.index
values=data['X1']
# values=list(values)
series = pd.Series(values,index=ind)
groups = series.groupby(pd.TimeGrouper('A'))
years = pd.DataFrame()
for name, group in groups:
	years[name.year] = group.values

years.boxplot()
years.plot(subplots=True, legend=False)
years = years.T
plt.matshow(years, interpolation=None, aspect='auto')
plt.show()
##################################

## MONTHLY PLOTS##################
series = pd.Series(data['X2'])
one_year = series['2015']
groups = one_year.groupby(pd.TimeGrouper('M'))
months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
months = pd.DataFrame(months)
months.columns = range(1,13)
# months.boxplot()
plt.matshow(months, interpolation=None, aspect='auto')
plt.show()
##################################

## TIME SERIES LAG SCATTER PLOTS#####
from pandas.tools.plotting import lag_plot
lag_plot(data['X2'],lag=1)
plt.show()

from pandas.tools.plotting import scatter_matrix
values = data['X1']
lags = 5
columns = [values]
for i in range(1,(lags + 1)):
	columns.append(values.shift(i))

dataframe = pd.concat(columns, axis=1)
columns = ['t+1']
for i in range(1,(lags + 1)):
	columns.append('t-' + str(i))

dataframe.columns = columns
plt.figure(1)
for i in range(1,(lags + 1)):
	ax = plt.subplot(240 + i)
	ax.set_title('t+1 vs t-' + str(i))
	plt.scatter(x=dataframe['t+1'].values, y=dataframe['t-'+str(i)].values)

plt.show()

# AUTO CORRELATION MATRIX
# series=pd.Series(np.diff(np.log(data['X1'])))
series=pd.Series(np.diff(data['X2']))
values = pd.DataFrame(series.values)
cols,names=[values],['t+1']
lags=1100
for i in range(1,lags):
	a=values.shift(i)
	cols.append(a)	
	b='t-'+str(i)
	names.append(b)

# [values.shift(7),values.shift(6),values.shift(5),values.shift(4),values.shift(3),values.shift(2),values.shift(1), values]
dataframe = pd.concat(cols, axis=1)
dataframe.columns = names
result = dataframe.corr()
print(result)
plt.plot(range(lags),result['t+1'])
plt.grid()
plt.show()

# TIME SERIES AUTO CORRELATION PLOTS
# NOT WORKING
from pandas.tools.plotting import autocorrelation_plot
series=pd.Series(np.diff(data['X2']))
autocorrelation_plot(series)
plt.show()

# AUTO CORRELATION PLOTS
data=data.fillna(1)
from statsmodels.graphics.tsaplots import plot_acf,plot_acf
series=pd.Series(np.diff(np.log(data['X1'])))
plot_acf(series,lags=400)
plot_pacf(series,lags=400)
plt.show()
# NOTE: X1 highly correlated with previous week day(0.72)

## STATIONARY CHECK###########FULLER TEST
from statsmodels.tsa.stattools import adfuller
series=np.diff(data['X2'])
X=series
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))