import numpy as np
import math
import pandas as pd

df=pd.read_csv("D:\Acad\PRML\Assignments\Assignment2\A2Q2Data_train.csv", sep=',',header=None)#The dataset is imported
#please change the location of the file to that in the local computer
dataset = df.to_numpy()

y=dataset[:,100]
X=dataset[:,0:100]


X=np.transpose(X)  #here the data points are in rows, this statement converts them to column notation
y=np.transpose(y)
w_ml=np.matmul(np.matmul((np.linalg.pinv(np.matmul(X,np.transpose(X)))),X),y)#The analytical solution for w_ml is directly used
print(w_ml)


