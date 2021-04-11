import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm=np.around(cm,decimals=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def visualize_confusion(Yts, Ypred, rel_list):
    """
    Plot confusion matrix
    """
    cnf_matrix = metrics.confusion_matrix(Yts, Ypred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(cnf_matrix,classes=rel_list,title='Confusion matrix, without normalization')
    # plt.savefig('abc.pdf', format='pdf')
    # Plot normalized confusion matrix
    plt.figure(figsize=(15,15))
    plot_confusion_matrix(cnf_matrix, classes=rel_list, normalize=True, title='Normalized confusion matrix')
    plt.show()

def generateX(data):
	X_num=data.iloc[:,range(2,12)+range(13,46)].values
	X_dum=data.iloc[:,[12]].values
	dfX_num=pd.DataFrame(X_num)
	fX_dum=pd.DataFrame(X_dum)
	dfX_dum=pd.get_dummies(fX_dum)
	dfX=pd.concat([dfX_num,dfX_dum],axis=1)
	return dfX

def generateY(data):
	Y=data.iloc[:,range(46,52)].values
	out=[]
	for i in range(40000):
		vector=Y[i]
		if vector[0]==1 and vector[3]==1:
			out.append('Supp')
		elif vector[1]==1 and vector[4]==1:
			out.append('Elite')
		elif vector[2]==1 and vector[5]==1:
			out.append('Credit')
		elif vector[0]==1 and vector[3]==0:
			out.append('NoInv')
		elif vector[1]==1 and vector[4]==0:
			out.append('NoInv')
		elif vector[2]==1 and vector[5]==0:
			out.append('NoInv')
	dfY=pd.DataFrame(out)
	return dfY

def train_predict(classifier):
	train_data=pd.read_csv("Training_Dataset.csv")
	test_data=pd.read_csv("Leaderboard_Dataset.csv")
	dfx=generateX(train_data)
	dfy=generateY(train_data)
	classes=list(np.unique(dfy[0]))
	dfx_test=generateX(test_data)
	model=classifier.fit(dfx,dfy.values.ravel())
	# Ypred = model.predict(dfx_test)
	Ypred_prob = model.predict_proba(dfx_test)
	return Ypred_prob,classes

def analyse(classifier):
	train_data=pd.read_csv("Training_Dataset.csv")
	dfx=generateX(train_data)
	dfy=generateY(train_data)
	classes=list(np.unique(dfy[0]))
	Xtr,Xts,Ytr,Yts = train_test_split(dfx,dfy,test_size = 0.3, random_state=0)
	model=classifier.fit(Xtr,Ytr.values.ravel())
	Ypred = model.predict(Xts)
	Ypred_prob = model.predict_proba(Xts)
	accuracy=metrics.accuracy_score(Ypred,Yts)
	print ("Accuracy of classifier: %f" % (accuracy))
	Yts_dum=pd.get_dummies(Yts)
	mse=metrics.mean_squared_error(Yts_dum.values,Ypred_prob)
	print ("Mean squared error: %f" % (mse))
	print(metrics.classification_report(Yts,Ypred,target_names=classes))
	visualize_confusion(Yts,Ypred,classes)

def generate_submission(pred,classes):
	test_data=pd.read_csv("Leaderboard_Dataset.csv")
	value=[]
	prob=[]
	for i in range(len(test_data)):
		a=pred[i][0]
		b=pred[i][1]
		c=pred[i][3]
		prob_list=[a,b,0,c]
		index=max(enumerate(prob_list),key=lambda x:x[1])[0]
		value.append(classes[index])
		prob.append(prob_list[index])		
	sub=pd.DataFrame(test_data['cm_key'])
	sub['ans']=value
	sub['prob']=prob
	# sub['ans'].value_counts()
	sub = sub.sort_values('prob', ascending=False)
	sub = sub.drop('prob', axis=1)
	sub=sub[0:1000]
	sub.to_csv("./result.csv", header=False, index=False)
	return sub

if __name__ == '__main__':
	#weight={'Supp': 5, 'Elite': 5, 'Credit': 5,'NoInv' :1}
	#forest=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0,class_weight=weight)
	#analyse(forest)
	forest=GradientBoostingClassifier(learning_rate=0.15,n_estimators=300,min_samples_split=850,min_samples_leaf=10,random_state=0)
	pred,classes=train_predict(forest)
	print('training done')
	final_submission=generate_submission(pred,classes)