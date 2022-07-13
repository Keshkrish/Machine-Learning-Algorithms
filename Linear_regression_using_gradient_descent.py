import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
epsilon=0.01# a tolerance is defined

def gradient(X,y,w):#gradient is calculated
    grad=((2*np.matmul(np.matmul(X,np.transpose(X)),w))-(2*np.matmul(X,y)))
    if np.linalg.norm(grad)>10:#When the norm is greater than 10, it is normalised so that large steps are avoided
        grad=grad/np.linalg.norm(grad)
    return grad


def gradient_descent(X,y,w_ml):#gradient descent is implemented
    w = np.random.rand(100)
    grad=gradient(X,y,w)
    difference=[]
    i=1.0
    while np.linalg.norm(grad)>epsilon:#the loop is exited when the gradient is less than a threshold
        grad = gradient(X, y, w)

        eta = 1/i
        w = w - (eta * grad)
        difference.append(np.linalg.norm(w - w_ml))#difference array stores the norm(w-w_ml) at the end of each iteration
        i = i + 1.0


    return w, difference

df=pd.read_csv("A2Q2Data_train.csv", sep=',',header=None)
dataset = df.to_numpy()

y=dataset[:,100]
X=dataset[:,0:100]


X=np.transpose(X)  #here the data points are in rows, this statement converts them to column notation

w_ml=np.matmul(np.matmul((np.linalg.pinv(np.matmul(X,np.transpose(X)))),X),y)#w_ml is calcuated analytically


w_gradient_descent,error=gradient_descent(X,y,w_ml)

print('The norm of w-w_ml after '+str(len(error))+' iterations is '+str(error[len(error)-1]))
print('The final value of w from gradient descent is :\n')
print(w_gradient_descent)



plt.plot(range(len(error)),error)
plt.title(' Plot of norm of w-w_ml vs iteration number for gradient descent')
plt.xlabel('iteration number(t)')
plt.ylabel('norm(w-w_ml)')
plt.show()
