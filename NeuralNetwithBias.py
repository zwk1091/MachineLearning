import numpy as np
# sigmod 函数, 当 deriv=true时返回sigmod函数的导数
def nonlin(x,deriv=False):
	if (deriv==True):
		return np.multiply(1-x,x)
	return 1/(1+np.exp(-x))


X=np.array([[1,0,0],[1,1,0],[1,0,1],[1,-1,0],[1,1,-1]])

y=np.array([[1],
			[1],
			[0],
			[0],
			[0]])

np.random.seed(1)
# 第一层权值 初始化为-1~1
syn0=2*np.random.random((3,2))-1
print("-----------------theta of layer 0")
print(syn0)
# 第二层权值
syn1=2*np.random.random((3,1))-1
print("-----------------theta of layer 1")
print(syn1)
#全部初始化为零
# syn0=np.zeros((3,2))
# syn1=np.zeros((3,1))
# l0=X
# l1=nonlin(np.dot(l0,syn0))
# l2=nonlin(np.dot(l1,syn1))

# dl1=nonlin(l1,deriv=True)
# dl2=nonlin(l2,deriv=True)
# print("-------deriv of l1 is ")
# print(dl1)
# print("-------deriv of l2 is ")
# print(dl2)
for j in range(5):
	print(syn0)
	print(syn1)
	l0=X
	l1=nonlin(np.dot(l0,syn0))
#add the bias to l1
	bias=np.mat(np.ones((5,1)))
	l1=np.hstack((bias,l1))
	# print("-----------------------l1 after adding bias")
	# print(l1)

	l2=nonlin(np.dot(l1,syn1))
	# print("-----------------------l2")
	# print(l2)
	l2_error=y-l2

	if(j%10000)==0:
		print ("Error:"+str(np.mean(np.abs(l2_error))))
	# print("-------------------l2_error is ")
	# print(l2_error.shape)
	# print("-------------------l2 is ")
	# print(l2.shape)
	# print("-------------------deriv of l2")
	# print(nonlin(l2,deriv=True))
	l2_delta=np.multiply(l2_error,nonlin(l2,deriv=True))

	# print("----------------syn1 T")
	# print(syn1.T.shape)
	# print("----------------l2 delta")
	# print(l2_delta.shape)

	l1_error=l2_delta.dot(syn1.T)

	# print("-------------------deriv of l1")
	# print(nonlin(l1,deriv=True))

	l1_delta=np.multiply(l1_error,nonlin(l1,deriv=True))

	syn1+=l1.T.dot(l2_delta)
	# print("---------------l1_delta")
	# print(l1_delta.shape)
	# print("---------------l0")
	# print(l0.shape)
	temp=l0.T.dot(l1_delta)
	syn0+=temp[:,1:5]

print ("Output hx After Training: ")
print (l2)

