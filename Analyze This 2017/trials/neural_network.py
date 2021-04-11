import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD

def generate_input(data):
	X_num=data.iloc[:,range(2,12)+range(13,46)].values
	X_dum=data.iloc[:,[12]].values
	dfX_num=pd.DataFrame(X_num)
	fX_dum=pd.DataFrame(X_dum)
	dfX_dum=pd.get_dummies(fX_dum)
	dfX=pd.concat([dfX_num,dfX_dum],axis=1)
	# Rescaling data points
	dfx=np.array(dfX)
	for i in (range(0,8)+range(9,42)):
		min_value=dfx[:,i].min()
		max_value=dfx[:,i].max()
		dfx[:,i]=(dfx[:,i]-min_value)/(max_value-min_value)
	return dfx

def pickle_write(data,filename):
    file_handle = open(filename, 'wb')
    pickle.dump(data,file_handle)
    file_handle.close()

def func(node,epoch,multi,file_number): 
	train_data=pd.read_csv("Training_Dataset.csv")
	Y=train_data.iloc[:,range(46,52)].values
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
			# out.append('No-Supp')
		elif vector[1]==1 and vector[4]==0:
			out.append('NoInv')
			# out.append('No-Elite')
		elif vector[2]==1 and vector[5]==0:
			out.append('NoInv')
			# out.append('No-Credit')

	dfY=pd.DataFrame(out)
	dfx=generate_input(train_data)
	X=np.array(dfx)
	classes=list(np.unique(dfY[0]))
	Y_dum=pd.get_dummies(dfY)
	y=np.array(Y_dum)

	model = Sequential()
	model.add(Dense(node, input_dim=61, activation='relu'))
	model.add(Dropout(0.5))
	if multi==True:
		model.add(Dense(61, activation='relu'))
		model.add(Dropout(0.5))
	model.add(Dense(4, activation='softmax'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
	              optimizer='sgd',
	              metrics=['accuracy'])

	store=model.fit(X, y, epochs=epoch, batch_size=32, verbose=0)

	test_data=pd.read_csv("Leaderboard_Dataset.csv")
	dfx=generate_input(test_data)
	X_test=np.array(dfx)
	y_pred=model.predict(X_test, batch_size=32)
	y_pred_classes=model.predict_classes(X_test, batch_size=32)

	value=[]
	prob=[]
	for i in range(10000):
		a=y_pred[i][0]
		b=y_pred[i][1]
		c=y_pred[i][3]
		prob_list=[a,b,0,c]
		index=max(enumerate(prob_list),key=lambda x:x[1])[0]
		value.append(classes[index])
		prob.append(prob_list[index])

	sub=pd.DataFrame(test_data['cm_key'])
	sub['ans']=value
	sub['prob']=prob
	sub['ans'].value_counts()
	sub = sub.sort_values('prob', ascending=False)
	sub = sub.drop('prob', axis=1)
	sub=sub[0:1000]

	weight=model.get_weights()
	filename='weight'+str(file_number)+'.pkl'
	pickle_write(weight,filename)
	hist=store.history
	filename='hist'+str(file_number)+'.pkl'
	pickle_write(hist,filename)
	filename='sub'+str(file_number)+'.csv'
	sub.to_csv(filename, header=False, index=False)

if __name__ == '__main__':
	print "case1...................................................."
	func(61,10000,False,1)
	print "case2...................................................."
	func(122,10000,False,2)
	print "case3...................................................."
	func(61,10000,True,3)
	print "case4...................................................."
	func(122,10000,True,4)
