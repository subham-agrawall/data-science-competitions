import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv('imputed_arimadata.csv', parse_dates=[0], index_col=0)
print(data.describe())
series=data['X1']

# from statsmodels.tsa.stattools import adfuller
# series=data['X2']
# X=series.values
# result = adfuller(np.diff(X))
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

# series=pd.Series(data['X2'])
dat=data.index[1:]
value=np.diff(data['X1'])
series=pd.Series(value,index=dat)
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series)
plt.show()
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
plot_acf(series,lags=25)
plot_pacf(series,lags=25)
plt.show()

## PERSISTENCE MODEL ###########################################
# create lagged dataset
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
n_test=89
train, test = X[1:len(X)-n_test], X[len(X)-n_test:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1] 
# persistence model
def model_persistence(x):
	return x

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)

test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()
################################################################

## AUTO_REGRESSION MODEL #######################################
from statsmodels.tsa.ar_model import AR
# split dataset
X = series.values
train, test = X[1:len(X)-n_test], X[len(X)-n_test:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error) #4789.437
# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
# X1- 12,263 to 4840
# X2-35797 to 70801
###############################################################

