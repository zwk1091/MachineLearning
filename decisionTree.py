import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
import math
TRAINNUM=5500
tmp = np.loadtxt("train.csv", dtype = np.str, delimiter =",")
X_Y = tmp[0:,1:]
X = X_Y[0:TRAINNUM, 0:128]
Y = X_Y[0:TRAINNUM, 128:]
print(X.shape)
X = X.astype(np.float64)
Y = Y.astype(np.int64)
y= Y.reshape([TRAINNUM])

 
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=10,min_samples_split=25)
regr_1.fit(X, y)

predic_tmp = np.loadtxt("test_raw.csv", dtype = np.str, delimiter =",")

# predict_X = predic_tmp[0:, 1:]
predict_X=X_Y[TRAINNUM:,0:128]
predict_X = predict_X.astype(np.float64)
true_Y= X_Y[TRAINNUM:, 128:]
# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]


#测试模型准确度
predict_Y = regr_1.predict(predict_X)
df = pd.read_csv('sampleSubmission.csv')
count=0
for i in range(500):
	df['Pred'][i] = predict_Y[i]
	if(df['Pred'][i]<1):
		df['Pred'][i]=1
	if(df['Pred'][i]>6):
		df['Pred'][i]=6
	df['Pred'][i]=int(round(df['Pred'][i]))
	# print(type(df['Pred'][i]))
	# print(int(true_Y[i][0]))
	# print(type(int(true_Y[i][0])))
	if(df['Pred'][i]==int(true_Y[i][0])):
		count=count+1
print(count)

#输出到文件
# df = pd.read_csv('sampleSubmission.csv')
# #先读取csv文件，将读取的内容保存下来，例如以list的形式保存，再对list进行修改。

# for i in range(7910):
# 	df['Pred'][i] = predict_Y[i]
# 	if(df['Pred'][i]<1):
# 		df['Pred'][i]=1
# 	if(df['Pred'][i]>6):
# 		df['Pred'][i]=6
# 	df['Pred'][i]=int(round(df['Pred'][i]))
# 	df['Pred'][i]= 'cls_'+str(df['Pred'][i])

# df.to_csv("dTree.csv",index= False, sep= ',')
