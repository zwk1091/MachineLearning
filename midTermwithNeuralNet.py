##这个版本在使用激活函数的时候有bug

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import pandas as pd 
import csv
import math

# sigmod 函数, 当 deriv=true时返回sigmod函数的导数
def nonlin(x,deriv=False):
	if (deriv==True):
		return np.multiply(1-x,x)
	# return longfloat(1/(1+np.exp(-x)))
	return 1/(1+np.exp(-x))

#读入数据的时候默认会忽略第一行。。。有点尬
trainSize=5999

data=pd.read_csv('train.csv')
# X=data(:,)
# print(data)
origin_X=data.iloc[:,1:129]

#进行归一化
min_max_scaler=MinMaxScaler()
# print(min_max_scaler)
X=min_max_scaler.fit_transform(origin_X)

labels=data.iloc[:,129]
y=np.zeros([trainSize,6])

for j in range(trainSize):
	temp=labels[j]
	y[j][temp-1]=1
# print(X)
# X=np.array(X).reshape(-1,1)
# print(y)
# print(y.shape)
# print(y)
# print("-----------------------------")
np.random.seed(1)
syn0=2*np.random.random((128,25))-1
# 输出层为6
syn1=2*np.random.random((25,6))-1

for j in range(5):
	l0=X
	# if j==1 :
	# 	print(syn0)
	# 	print(syn1)

	l1=nonlin(np.dot(l0,syn0))

	l2=nonlin(np.dot(l1,syn1))
	
	# print(l2.shape)
	if  j==2:
		print("------------dot of l0 and syn0")
		print(np.dot(l0,syn0))

	l2_error=y-l2
	# if j==0 :
	# 	print("-----------l2_ Error is ")
	# 	print(l2_error)
	# if(j%10000)==0:
	# print ("Error:"+str(np.mean(np.abs(l2_error))))
	
	l2_delta=l2_error*nonlin(l2,deriv=True)
	# print("-----------l2_ delta is ")
	# print(l2_delta)

	l1_error=l2_delta.dot(syn1.T)

	l1_delta=l1_error*nonlin(l1,deriv=True)

	if  j==0:
		print("------------dot of l1_delta")
		print(l1_delta)
		print("----------l0")
		print(l0)
		print("-----------l0.T.dot(l1_delta)")
		print(l0.T.dot(l1_delta))

	syn1+=l1.T.dot(l2_delta)
	syn0+=l0.T.dot(l1_delta)

print ("Output hx After Training: ")
print("------------------l2 shape")
print (l2.shape)
print(l2)
# test_raw=pd.read_csv('test_raw.csv')
# # print(test_raw.iloc[1,:])

# with open('testWrite.csv','w',newline='') as csvfile:

# 	writer=csv.writer(csvfile)
# 	writer.writerow(['ID','Pred'])
# 	writer.writerow(['ID_00006000','cls_3'])
# 	for num in range(7909) : 
# 		test1=test_raw.iloc[num,1:129]
# 		# if num==2:
# 		strIndex=test_raw.iloc[num,0]
# 		strRes="cls_"+str(np.rint(model.predict(np.array(test1).reshape(1,-1)))[0])[0]
# 			# print(strIndex)
# 		writer.writerow([strIndex,strRes])

# print(model.predict(np.array(test1).reshape(1,-1)))
# print(model.predict(test1))
