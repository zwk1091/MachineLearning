##0.99157 by ExtraTreesClassifier
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import math
TRAINNUM=6000
tmp = np.loadtxt("train.csv", dtype = np.str, delimiter =",")
X_Y = tmp[0:,1:]
X = X_Y[0:TRAINNUM, 0:128]
Y = X_Y[0:TRAINNUM, 128:]
print(X.shape)
X = X.astype(np.float64)
Y = Y.astype(np.int64)
y= Y.reshape([TRAINNUM])

parameters={
	'n_estimators':(49,50,60,70),
	'max_features':(11,12,13),
	# 'min_samples_split':(2,3,4,5,6,7,8,9,10,15,20)
	'min_samples_split':(2,3)
}

#自动参数选取
# pipeline=Pipeline(['z',ExtraTreesClassifier(criterion='entropy')])
clf=ExtraTreesClassifier()

# grid_search=GridSearchCV(clf,parameters,cv=5,scoring='accuracy')
# Fit regression model

# regr_1 = DecisionTreeRegressor(max_depth=10,min_samples_split=25)
# regr_1=RandomForestClassifier(n_estimators=10,min_samples_split=10)
grid_search=ExtraTreesClassifier(n_estimators=60,max_features=12,min_samples_split=2)
# regr_1=BaggingClassifier(n_estimators=15,max_samples=0.6)

# regr_1.fit(X, y)
grid_search.fit(X,y)
# print('最佳效果' % grid_search.best_score_)
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print('\t%s: %r' % (param_name, best_parameters[param_name]))

predic_tmp = np.loadtxt("test_raw.csv", dtype = np.str, delimiter =",")

predict_X = predic_tmp[0:, 1:]
# predict_X=X_Y[TRAINNUM:,0:128]
predict_X = predict_X.astype(np.float64)
true_Y= X_Y[TRAINNUM:, 128:]
# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

predict_Y = grid_search.predict(predict_X)

#测试模型准确度
# df = pd.read_csv('sampleSubmission.csv')
# count=0
# for i in range(6000-TRAINNUM):
# 	df['Pred'][i] = predict_Y[i]
# 	if(df['Pred'][i]<1):
# 		df['Pred'][i]=1
# 	if(df['Pred'][i]>6):
# 		df['Pred'][i]=6
# 	df['Pred'][i]=int(round(df['Pred'][i]))
# 	# print(type(df['Pred'][i]))
# 	# print(int(true_Y[i][0]))
# 	# print(type(int(true_Y[i][0])))
# 	if(df['Pred'][i]==int(true_Y[i][0])):
# 		count=count+1
# print(count)

# 输出到文件
df = pd.read_csv('sampleSubmission.csv')
#先读取csv文件，将读取的内容保存下来，例如以list的形式保存，再对list进行修改。

for i in range(7910):
	df['Pred'][i] = predict_Y[i]
	if(df['Pred'][i]<1):
		df['Pred'][i]=1
	if(df['Pred'][i]>6):
		df['Pred'][i]=6
	df['Pred'][i]=int(round(df['Pred'][i]))
	df['Pred'][i]= 'cls_'+str(df['Pred'][i])

df.to_csv("RandomForest.csv",index= False, sep= ',')