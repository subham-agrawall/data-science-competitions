# Voting Ensemble for Classification
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

data=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
X=data.iloc[:,range(2,28)].values
y=data.iloc[:,1].values
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

# create the sub models
estimators = []
model1 = LogisticRegressionCV(class_weight='balanced')
estimators.append(('logistic', model1))
model2 = XGBClassifier(learning_rate =0.03,
	n_estimators=100,
	max_depth=4,
	min_child_weight=6,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=1,
	seed=27)
estimators.append(('xgb', model2))
model3 = GaussianNB()
estimators.append(('naive', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators,voting='soft',n_jobs=-1)
results = model_selection.cross_val_score(ensemble, X, y, cv=kfold,scoring='roc_auc')
print(results.mean())