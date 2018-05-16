# for x in range(10):
# 	print(x)
# import numpy as np 
# import pandas as pd 
# t=100
# y=np.zeros([t,2])
# print(y)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
X_train = np.array([[ 1., -1.,  2.], 
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler=MinMaxScaler()

print(min_max_scaler)

X_train_minmax=min_max_scaler.fit_transform(X_train)
print(X_train_minmax)
