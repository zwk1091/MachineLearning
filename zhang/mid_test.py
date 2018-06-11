#data mining mid test python
#id 128 lables


import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# 6000 samples, and 128 features and a lable 1-6
tmp = np.loadtxt("train.csv", dtype = np.str, delimiter =",")
X_Y = tmp[0:,1:]
X = X_Y[0:, 0:128]
Y = X_Y[0:, 128:]

X = X.astype(np.float64)
Y = Y.astype(np.float64)

regr = linear_model.LinearRegression()
regr.fit(X,Y)
regr.score(X,Y)
print(regr.score(X,Y))

a,b = regr.coef_, regr.intercept_

predic_tmp = np.loadtxt("test_raw.csv", dtype = np.str, delimiter =",")
predict_X = predic_tmp[0:, 1:]
predict_X = predict_X.astype(np.float64)

#predict_Y = np.dot(predict_X,a.T) +b
predict_Y = regr.predict(predict_X)

predict_Y = predict_Y.astype(np.int64)

print(predict_Y)
print(predict_Y.shape)


# 读取csv至字典
df = pd.read_csv('sampleSubmission.csv')
#先读取csv文件，将读取的内容保存下来，例如以list的形式保存，再对list进行修改。
df['Pred'] = predict_Y

for i in range(7910):
	if(df['Pred'][i]<1):
		df['Pred'][i]=1
	if(df['Pred'][i]>6):
		df['Pred'][i]=6
	df['Pred'][i]= 'cls_'+str(df['Pred'][i])




df.to_csv("result1.csv",index= False, sep= ',')