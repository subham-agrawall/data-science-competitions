from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced

RANDOM_STATE = 42

data=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
X=data.iloc[:,range(2,28)].values
y=data.iloc[:,1].values

pipeline = pl.make_pipeline(os.SMOTE(random_state=RANDOM_STATE),
                            SVC(random_state=RANDOM_STATE))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=RANDOM_STATE)

# Train the classifier with balancing
pipeline.fit(X_train, y_train)

# Test the classifier and get the prediction
y_pred_bal = pipeline.predict(X_test)

# Show the classification report
from sklearn import metrics
accuracy=metrics.roc_auc_score(y_test, y_pred_bal)
print(accuracy)
print(classification_report_imbalanced(y_test, y_pred_bal))