import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def k_radial(x,y,sigma): #defining radial kernel
    z=np.dot(np.subtract(x,y),np.subtract(x,y))
    z=-z/(2*sigma*sigma)
    return math.exp(z)
def k_poly(x,y,case): #defining polynomial kernel
    z=(np.dot(x,y)+1)**case
    return z



def Kernel_poly(input_data,case): #obtaining K matrix for polynomial
    K=np.zeros([1000,1000])
    for i in range(1000):
        for j in range(1000):
            K[i,j]=k_poly(input_data[:,i],input_data[:,j],case)



    return K


def Kernal_radial(input_data,sigma): #Obtaining K matrix for radial
    K=np.zeros([1000,1000])
    for i in range(1000):
        for j in range(1000):
            K[i,j]= k_radial(input_data[:,i],input_data[:,j],sigma)

    return K

def Kernel_centering(K): #A function to center the Kernel matrix
    Z = np.identity(1000) - (np.ones((1000, 1000)) / 1000)
    K_centered = np.matmul(np.matmul(Z, K), Z) #done in matrix form instead of element wise
    return K_centered

def PCA(input_data,type,case): #does kernel PCA on the given dataset, type determines the type of the kernel function, if type=1 then polynomial; if type =2 then radial
    if type==1:                #case is the value of d and sigma for polynomial and radial respectively

        K=Kernel_poly(input_data,case) #computing the kernel matrix
        K=Kernel_centering(K)         #centering the kernel matrix
        eigenval,eigenvec=np.linalg.eig(K) #computing the eigenvectors of K
        idx=eigenval.argsort()[::-1]     #sorting the eigenvectors in the decreasing order of the eigen values
        eigenval=eigenval[idx]
        eigenvec=eigenvec[:,idx]
        for i in range(2):
            eigenvec[:,i]=eigenvec[:,i]/math.sqrt(eigenval[i])  #eigen vectors are normalised such that the principal components has unit length
                                                                #the eigenvectors after normalising are the alphas, only the first 2 eigen vectors are normalised as only they are required as per the question


        #print('For polynomial kernel with d=',case)
        #print('\n')
        #print('Alpha1 is:\n',eigenvec[:,0])
        #print('Alpha2 is :\n',eigenvec[:,1])
        #uncomment the above statements to print the value of alpha1 and alpha2

        projection_1=np.matmul(K,eigenvec[:,0]) #projection on a principal component is the dot product of alpha and k(xi,:), k(xi,:) is the ith row of K; writing it in matrix form we get this
        projection_2=np.matmul(K,eigenvec[:,1])

        #print('Projection of data points on first principal component is:\n',projection_1)
        #print('Projection of data points on second principal component is:\n',projection_2)
        #uncomment the above statements to print the projections of the data points on the first and second principal components

        variance1=np.sum(np.square(projection_1))*0.001 #variance of the data on first principal component
        variance2=np.sum(np.square(projection_2))*0.001 #variance of the data on second principal component
        total_variance=variance1+variance2 #variance of the data on the first 2 components
        print('The variance of the data on the top 2 principal components for polynomial kernel with d='+str(case)+' is '+str(total_variance))
        plt.plot(projection_1, projection_2, 'ro') #the projections are plotted
        plt.title('Polynomial Kernel with d ='+str(case))
        plt.xlabel('Projection onto the first principal component')
        plt.ylabel('Projection onto the second principle component ')
        plt.show()

    if type==2:
        K=Kernal_radial(input_data,case)
        K = Kernel_centering(K)
        eigenval,eigenvec=np.linalg.eig(K)
        idx=eigenval.argsort()[::-1]
        eigenval=eigenval[idx]
        eigenvec=eigenvec[:,idx]
        for i in range(2):
            eigenvec[:,i]=eigenvec[:,i]/math.sqrt(eigenval[i])

        #print('For radial kernel with d=', case)
        #print('\n')
        #print('Alpha1 is:\n', eigenvec[:, 0])
        #print('\nAlpha2 is :\n', eigenvec[:, 1])
        #uncomment above statements for alpha
        projection_1=np.matmul(K,eigenvec[:,0])
        projection_2=np.matmul(K,eigenvec[:,1])
        #print('\nProjection of data points on first principal component is:\n', projection_1)
        #print('\nProjection of data points on second principal component is:\n', projection_2)
        #uncomment above statements for projections
        variance1 = (np.sum(np.square(projection_1)))*0.001  # variance of the data on first principal component
        variance2 = (np.sum(np.square(projection_2)))*0.001  # variance of the data on second principal component
        total_variance = variance1 + variance2  # variance of the data on the first 2 components
        print('The variance of the data on the top 2 principal components for radial kernel with sigma=' + str(case) + ' is ' + str(total_variance))
        plt.plot(projection_1, projection_2, 'ro')
        plt.title('Radial Kernel with sigma =' + str(case))
        plt.xlabel('Projection onto the first principal component')
        plt.ylabel('Projection onto the second principle component ')
        plt.show()


df=pd.read_csv('D:\Acad\PRML\Assignments\Assignment1\Dataset.csv', sep=',',header=None)
dataset = df.to_numpy()
X=np.transpose(dataset)

for i in range(2,4,1):
    PCA(X,1,i)
sigma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i in sigma:
    PCA(X,2,i)

