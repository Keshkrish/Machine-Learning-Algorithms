import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

epsilon=0.01

def gradient(X,y,w):
    grad=((2*np.matmul(np.matmul(X,np.transpose(X)),w))-(2*np.matmul(X,y)))
    if np.linalg.norm(grad)>10:
        grad=grad/np.linalg.norm(grad)
    return grad

def stochastic_gradient_descent(X,y,w_ml):
    w = np.random.rand(100)
    difference = []
    j = 1.0
    grad=gradient(X,y,w)

    while np.linalg.norm(grad)>epsilon:
        batch = np.random.choice(10000, 100, replace=False)#The batch is chosen
        X_ = np.zeros((100, 100))
        for i in range(100):
            X_[:, i] = X[:, batch[i]]

        y_ = []
        for i in batch:
            y_.append(y[i])

        grad = gradient(X_, y_, w)#Gradient is computed using this batch
        eta = 1/j
        w = w - (eta * grad)
        difference.append(np.linalg.norm(w-w_ml))
        j=j+1
        if j>50000:#the algorithm stops if number of iterations exceed 50000

            return w,difference


    return w,difference


df=pd.read_csv("D:\Acad\PRML\Assignments\Assignment2\A2Q2Data_train.csv", sep=',',header=None)
dataset = df.to_numpy()

y=dataset[:,100]
X=dataset[:,0:100]


X=np.transpose(X)  #here the data points are in rows, this statement converts them to column notation

w_ml=np.matmul(np.matmul((np.linalg.pinv(np.matmul(X,np.transpose(X)))),X),y)


w_stoc,error=stochastic_gradient_descent(X,y,w_ml)


print('The norm of w-w_ml after '+str(len(error))+' iterations is '+str(error[len(error)-1]))
print('The final value of w from gradient descent is :\n')
print(w_stoc)


plt.plot(range(len(error)),error)
plt.title('Norm of w-w_ml vs iteration number for stochastic gradient descent')
plt.xlabel('iteration number(t)')
plt.ylabel('norm(w-w_ml)')
plt.show()