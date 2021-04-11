import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

data = pd.read_excel('TestSet_IITM.xlsx', parse_dates=[0], index_col=0)
print(data.describe())

# Create lagged dataset
series=pd.Series(data['X2'])
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))
 
X = dataframe.values
test_X, test_y = X[1:,0], X[1:,1]
 
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
# BASELINE MODELS
## MSE=10270.657(X1)
## MSE=15545.036(X2)

# plot predictions and expected results
plt.plot(test_y)
plt.plot(predictions)
plt.show()