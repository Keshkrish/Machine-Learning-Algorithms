import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

epsilon=0.01
def gradient(X,y,w,lam):#The gradient is modified to accomodate the lambda*w term
    grad=((2*np.matmul(np.matmul(X,np.transpose(X)),w))-(2*np.matmul(X,y))+lam*w)
    if np.linalg.norm(grad)>10:
        grad=grad/np.linalg.norm(grad)
    return grad
def gradient_descent(X,y,lam):#Same as 2.ii with the modification for lambda

    w=np.random.rand(100)
    grad=gradient(X,y,w,lam)
    i=1.0

    while np.linalg.norm(grad)>epsilon:
        grad=gradient(X,y,w,lam)

        eta=1/i
        w=w-(eta*grad)

        i=i+1.0

    return w

df=pd.read_csv("D:\Acad\PRML\Assignments\Assignment2\A2Q2Data_train.csv", sep=',',header=None)
dataset = df.to_numpy()

y=dataset[:,100]
X=dataset[:,0:100]


X=np.transpose(X)  #here the data points are in rows, this statement converts them to column notation

w_ml=np.matmul(np.matmul((np.linalg.pinv(np.matmul(X,np.transpose(X)))),X),y)#w_ml is calculated analytically

lam_values=range(10)#we do cross validation to find the best value of lambda
#We run the algorithm for 10 lambdas(0,1,2...,9)
X_train=np.zeros((100,8000))#The data set is split into training and validation set
y_train=[]
for i in range(8000):
    X_train[:,i]=X[:,i]
    y_train.append(y[i])

X_validate=np.zeros((100,2000))
y_validate=[]
for i in range(2000):
    X_validate[:,i]=X[:,8000+i]
    y_validate.append(y[8000+i])

error_validation=[]#stores the error in the validation set for each lamda

for lam in lam_values:
    w = gradient_descent(X_train, y_train,  lam)


    y_pred = np.matmul(np.transpose(X_validate), w)
    error = (np.linalg.norm(y_pred - y_validate)) ** 2

    error_validation.append(error)




best_lam_index=np.argmin(error_validation)
best_lam=lam_values[best_lam_index]#the best lamda is chosen as the one that gives minimum error in the validation set

plt.plot(lam_values,error_validation)
plt.title('Error in validation set vs lambda')
plt.xlabel('Lambda')
plt.ylabel('Error in validation set')
plt.show()
A=np.matmul(X,np.transpose(X))
I=np.identity(len(A))
w_r=np.matmul(np.matmul(np.linalg.pinv(A+(best_lam*I)),X),y)#After finding the best lambda using gradient descent, w_r is computed analytically using this lambda

df2=pd.read_csv("D:\Acad\PRML\Assignments\Assignment2\A2Q2Data_test.csv", sep=',',header=None)#test set is imported
dataset2 = df2.to_numpy()

y_test=dataset2[:,100]
X_test=dataset2[:,0:100]
X_test=np.transpose(X_test)

y_pred_r=np.matmul(np.transpose(X_test),w_r)
y_pred_ml=np.matmul(np.transpose(X_test),w_ml)

error_r = (np.linalg.norm(y_pred_r - y_test)) ** 2#performance of ridge regression and linear regression is measured
error_ml=(np.linalg.norm(y_pred_ml-y_test))**2
print("The error in test data by the ridge regression predictor is:",error_r)
print("The error in test data by the linear regression predictor is:",error_ml)
















