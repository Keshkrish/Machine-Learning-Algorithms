import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


#All the functions below are the same as the ones in Q2.i and Q1.iii
def k_poly(x,y,case):
    z=(np.dot(x,y)+1)**case
    return z
def k_radial(x,y,sigma): #defining radial kernel
    z=np.dot(np.subtract(x,y),np.subtract(x,y))
    z=-z/(2*sigma*sigma)
    return math.exp(z)

def Kernel_poly(input_data,case):
    K=np.zeros([1000,1000])
    for i in range(1000):
        for j in range(1000):
            K[i,j]=k_poly(input_data[:,i],input_data[:,j],case)



    return K
def Kernal_radial(input_data,sigma):
    K=np.zeros([1000,1000])
    for i in range(1000):
        for j in range(1000):
            K[i,j]= k_radial(input_data[:,i],input_data[:,j],sigma)

    return K

def means(input_data,clusters,no_cluster,d):
    mean_cluster=np.zeros([d,no_cluster ])
    count=np.zeros(no_cluster)
    for i in range(1000):
        mean_cluster[:,clusters[i]]+=input_data[:,i]
        count[clusters[i]]+=1.0
    for i in range(no_cluster):
        mean_cluster[:, i] = mean_cluster[:, i] * (1 / count[i])

    return mean_cluster
def kmeans(input_data,no_clusters):
    k_means=np.random.choice(1000,no_clusters,replace=False)
    z=np.zeros(1000,dtype=int)
    for i in range(1000):
        dist=np.zeros(no_clusters)
        for j in range(no_clusters):
            dist[j]=np.dot(input_data[:,i]-input_data[:,k_means[j]],input_data[:,i]-input_data[:,k_means[j]])

        z[i]=np.argmin(dist)

    return z

def Lloyds(input_data,initial_cluster,no_cluster,d):

    z_current=initial_cluster
    z_next=z_current


    n_iteration=0
    while(1):

        z_current=z_next
        z_next=np.zeros(1000,dtype=int)

        mean=means(input_data,z_current,no_cluster,d)

        for i in range(1000):
            dist_array=np.zeros(no_cluster)
            for j in range(no_cluster):
                dist_array[j]=np.dot(input_data[:,i]-mean[:,j],input_data[:,i]-mean[:,j])

            z=np.argmin(dist_array)
            if np.dot(input_data[:,i]-mean[:,z],input_data[:,i]-mean[:,z]) < np.dot(input_data[:,i]-mean[:,z_current[i]],input_data[:,i]-mean[:,z_current[i]]):
                z_next[i]=z
            else:
                z_next[i]=z_current[i]



        if np.array_equal(z_current,z_next):
            return z_next
        n_iteration+=1




df=pd.read_csv('D:\Acad\PRML\Assignments\Assignment1\Dataset.csv', sep=',',header=None)
dataset = df.to_numpy()
X=np.transpose(dataset)

#K=Kernel_poly(X,3)
K=Kernal_radial(X,0.1) #Radial basis function with sigma=0.1 is used.#remove this line and uncomment the above line for polynomial kernel
eigenval, eigenvec = np.linalg.eig(K)
idx = eigenval.argsort()[::-1]
eigenval = eigenval[idx]
eigenvec = eigenvec[:, idx]

H=np.zeros([1000,4])

for i in range(4):
    H[:,i]=eigenvec[:,i]#H matrix is defined. Its columns are the top k eigen vectors of K matrix


for i in range(1000):
    H[i,:]=H[i,:]*(1.0/np.linalg.norm(H[i,:]))#The rows of H are normalised to unit length



#Now we perform Kmeans on the rows of the H matrix
#First we take its transpose so that the data set matches our usual notation of dxn

H=np.transpose(H)

z=kmeans(H,4)
clust=Lloyds(H,z,4,4)

for i in range(1000):  # plotting the data points
    if clust[i] == 0:
        plt.plot(X[0, i], X[1, i], 'ro')  # red for cluster 1
    elif clust[i] == 1:
        plt.plot(X[0, i], X[1, i], 'bo')  # blue for cluster 2
    elif clust[i] == 2:
        plt.plot(X[0, i], X[1, i], 'go')  # green for cluster 3
    elif clust[i] == 3:
        plt.plot(X[0, i], X[1, i], 'yo')  # yellow for cluster 4

plt.title('Plot of the data points with clusters in mapped space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
