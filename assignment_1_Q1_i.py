import numpy as np
import math
import pandas as pd

def PCA(input_data):
    XT=np.transpose(input_data)
    X=input_data
    C= (np.matmul(X,XT))*0.001 # Covariance matrix = 1/n(X*XT)

    eigval,eigvec=np.linalg.eig(C) #Eigen values and eigen vectors of C, Principal components are the eigenvectors in descending order according to eigen values
    idx = eigval.argsort()[::-1] # Eigenvalues and corresponding eigenvectors are ordered decending
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    variance1=0.0 #variance along 1st compinent
    variance2=0.0 #varaince along 2nd component
    for i in range(1000):
        variance1+=np.dot(X[:,i],eigvec[:,0])**2
        variance2+=np.dot(X[:,i],eigvec[:,1])**2
    variance1=variance1/1000.0
    variance2=variance2/1000.0

    return eigval,eigvec,variance1,variance2




df=pd.read_csv('D:\Acad\PRML\Assignments\Assignment1\Dataset.csv', sep=',',header=None)
dataset = df.to_numpy()


X=np.transpose(dataset)

col_sum=X.sum(axis=1) #computing the mean
col_mean=col_sum*0.001

print('The mean of the data is:',col_mean)
#centering the data
for i in range(1000):
    X[:,i]=X[:,i]-col_mean


eigval,eigvec,variance1,variance2=PCA(X)
print('The eigenvalues are:',eigval)
print('The principal components are:\n',eigvec)
print('The variance along 1st principal component:',variance1)
print('The variance along 2nd principal component:',variance2)
