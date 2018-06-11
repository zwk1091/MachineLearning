# tensoflow a easy network to classify 
import csv
import pandas as pd
import numpy as np
from numpy import argmax
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model,svm,tree
from sklearn import multiclass
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

tmp = np.loadtxt("train.csv", dtype = np.str, delimiter =",")
X_Y = tmp[0:,1:]
X = X_Y[0:, 0:128]
Y = X_Y[0:, 128:]

X = X.astype(np.float64)
Y = Y.astype(np.int64)

# noramlization

# oneHot Y 
# use to compare with softmax

Y=Y.reshape((6000,))
label_encoder = LabelEncoder()
# input [size,] output[size,]
integer_encoded = label_encoder.fit_transform(Y)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
OneHot_encoder = OneHotEncoder(sparse = False)
#input[6000,1] output[6000, n_class]
OneHot_Y = OneHot_encoder.fit_transform(integer_encoded)

print(Y)
print(OneHot_Y)
print(OneHot_Y.shape)

print(argmax(OneHot_Y[0,:]))

#inverse_transform 输入为[6000,n_class] [6000,1]
#inverted_1 = OneHot_encoder.inverse_transform(OneHot_Y)
#inverted_2 = label_encoder.inverse_transform(inverted_1)
#print(inverted_1.shape)
#print(inverted_2.shape)

predict_tmp = np.loadtxt("test_raw.csv", dtype = np.str, delimiter =",")
predict_X = predict_tmp[0:, 1:]
predict_X = predict_X.astype(np.float64)

## X [6000,128]
## OneHot_Y [6000,6]
## predict_X [7910,128]
## predict_Y [7910,6 ]



# softmax
#-------------------
x = tf.placeholder("float", shape=[None, 128])
y_ = tf.placeholder("float", shape=[None, 6])












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




df.to_csv("result_softmax.csv",index= False, sep= ',')



