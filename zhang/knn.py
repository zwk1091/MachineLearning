import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
from sklearn import datasets, linear_model,svm,tree
from sklearn import multiclass

# 6000 samples, and 128 features and a lable 1-6
tmp = np.loadtxt("../train.csv", dtype = np.str, delimiter =",")
X_Y = tmp[0:,1:]
X = X_Y[0:, 0:128]
Y = X_Y[0:, 128:]

X = X.astype(np.float64)
Y = Y.astype(np.int64)



Y= Y.reshape([6000])

predic_tmp = np.loadtxt("../test_raw.csv", dtype = np.str, delimiter =",")
predict_X = predic_tmp[0:, 1:]
predict_X = predict_X.astype(np.float64)


#进行归一化
# print(X)

# scaler=StandardScaler()
# X=scaler.fit_transform(X)
# predict_X=scaler.fit_transform(predict_X)
# X_=X
print(X)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X,Y)
print(knn.score(X,Y))
predict_Y = knn.predict(predict_X)

print(predict_Y)

# 读取csv至字典
df = pd.read_csv('../sampleSubmission.csv')
#先读取csv文件，将读取的内容保存下来，例如以list的形式保存，再对list进行修改。


for i in range(7910):
	df['Pred'][i] = predict_Y[i]
	if(df['Pred'][i]<1):
		df['Pred'][i]=1
	if(df['Pred'][i]>6):
		df['Pred'][i]=6
	df['Pred'][i]= 'cls_'+str(df['Pred'][i])




df.to_csv("knn1.csv",index= False, sep= ',')