import pandas as pd
import matplotlib as plt
import numpy as np
import string
import os
from pandas import DataFrame as df

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

#read data
pag_with_dummy = df(pd.read_csv('pag_with_dummy.txt'))

#divide data into three groups: za xq xz
za_final = pag_with_dummy[pag_with_dummy['label'] == 2]
xz_final = pag_with_dummy[pag_with_dummy['label'] == 1]
xq_final = pag_with_dummy[pag_with_dummy['label'] == 0]

#za_final.shape[0]=600044,xz_final.shape[0]=101867,xq_final.shape[0]=26531
za = za_final.sample(n=100000,random_state=333)
za_xz = pd.concat([za,xz_final],axis=0) #total_numbers=201867
print(za_xz.columns.tolist())

za_xz = za_xz.reset_index(drop=True)

za1 = za_final.sample(n=28000,random_state=334)
za_xq = pd.concat([za1,xq_final],axis=0) #total_numbers=54531
#za_xq = za_xq.reset_index()
xz = xz_final.sample(n=28000,random_state=334)
xz_xq = pd.concat([xz,xq_final],axis=0) #total_numbers=54531
#xz_xq = xz_xq.reset_index()
#define train process
def pro(data):
	y = data['label']
	X = data.drop(['label','device_id'],axis=1)
	ss=StandardScaler()
	X_regular=ss.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=333)
	lr = LogisticRegression(C=1.0, class_weight='balanced',dual=False, fit_intercept=True,n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',tol=0.0001, verbose=0, warm_start=False)
	lr.fit(X_train, y_train)
	y_predict = lr.predict(X_test)
	print('Accuracy of LR Classifier:', lr.score(X_test,y_test))
	print(classification_report(y_test, y_predict))
	weight = lr.coef_
	print('weight:',weight)
	return data

print('za_xz train')
za_xz.apply(pro,axis=0,reduce=False)

print('za_xq train')
#za_xq.apply(pro,axis=0)

print('xz_xq train')
#xz_xq.apply(pro,axis=0)
