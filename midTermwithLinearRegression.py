from sklearn.linear_model import LinearRegression 
import numpy as np 
import pandas as pd 
import csv
import math

data=pd.read_csv('train.csv')
# X=data(:,)
# print(data)
X=data.iloc[:,1:129]
y=data.iloc[:,129]
# print(X)
# X=np.array(X).reshape(-1,1)
# print(y)
model =LinearRegression()

model.fit(X,y)

test_raw=pd.read_csv('test_raw.csv')
# print(test_raw.iloc[1,:])

with open('testWrite.csv','w',newline='') as csvfile:

	writer=csv.writer(csvfile)
	writer.writerow(['ID','Pred'])
	writer.writerow(['ID_00006000','cls_3'])
	for num in range(7909) : 
		test1=test_raw.iloc[num,1:129]
		# if num==2:
		strIndex=test_raw.iloc[num,0]
		strRes="cls_"+str(np.rint(model.predict(np.array(test1).reshape(1,-1)))[0])[0]
			# print(strIndex)
		writer.writerow([strIndex,strRes])

# print(model.predict(np.array(test1).reshape(1,-1)))
# print(model.predict(test1))
