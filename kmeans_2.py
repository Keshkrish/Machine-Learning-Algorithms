import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

'''The functions defined below are the same as the ones I defined in assignment 1 Q2.1'''
def means(input_data,clusters,no_cluster,d): # This function takes the input data and the clusters(along with number of clusters and dimension of data points d) and computes the means of each cluster
    mean_cluster=np.zeros([d,no_cluster ])
    count=np.zeros(no_cluster)
    for i in range(1000):
        mean_cluster[:,clusters[i]]+=input_data[:,i]
        count[clusters[i]]+=1.0
    for i in range(no_cluster):
        mean_cluster[:, i] = mean_cluster[:, i] * (1 / count[i])

    return mean_cluster


def error(initial_data,cluster,mean): #A function that calculates the error at the end off each iteration
    err=0.0
    for i in range(1000):
        err+=np.dot(initial_data[:,i]-mean[:,cluster[i]],initial_data[:,i]-mean[:,cluster[i]])
    return err


def Lloyds(input_data,initial_cluster,no_cluster,d): #Lloyds algorithm for clustering
#no_clusters-> number of clusters
#d->dimension of data points

    z_current=initial_cluster #z_current[i] contains the cluster to which the ith data point belongs to in the current iteration
    z_next=z_current #z_next contains the cluster numbers for data points after current iteration
                    #it is initialized to z_current so that the value of z_current does not change due to the first statement in the first iteration of the following while loop

    error_array=[]#an array that stores the value of error at each iteration


    n_iteration=0#stores the iteration number

    while(1):

        z_current=z_next #the value of z_current becomes z_next before the beggining of an iteration(This statement does not change anything in the first iteration as z_next=z_current initially)
        z_next=np.zeros(1000,dtype=int) #z_next is initialized

        mean=means(input_data,z_current,no_cluster,d) #mean of the clusters are computed
        error_cluster=error(input_data,z_current,mean)#the error at current iteration is computed and stored
        error_array=np.append(error_array,error_cluster)

        for i in range(1000):
            dist_array=np.zeros(no_cluster)
            for j in range(no_cluster):#dist array stores the distance of the data points from the means
                dist_array[j]=np.dot(input_data[:,i]-mean[:,j],input_data[:,i]-mean[:,j])

            z=np.argmin(dist_array)#the cluster that gives the lowest distance is obtained
            if np.dot(input_data[:,i]-mean[:,z],input_data[:,i]-mean[:,z]) < np.dot(input_data[:,i]-mean[:,z_current[i]],input_data[:,i]-mean[:,z_current[i]]):
                z_next[i]=z
            else:
                z_next[i]=z_current[i]
            #the above if else statement ensures that the cluster is updated only if the distance from the new cluster is strictly less than the current cluster

        n_iteration += 1
        if np.array_equal(z_current,z_next):# to check if convergence has occured and terminate the function

            plt.plot(np.arange(n_iteration),error_array)#error is plotted
            plt.title('Plot of error vs number of iterations')
            plt.xlabel('number of iterations')
            plt.ylabel('error function')
            plt.show()
            return z_next,mean




def kmeans(input_data,no_clusters): #a function to do kmeans initialization
    k_means=np.random.choice(1000,no_clusters,replace=False)#picks k random indices whose data points becomes the k means
    z=np.zeros(1000,dtype=int)
    for i in range(1000):
        dist=np.zeros(no_clusters)
        for j in range(no_clusters):
            dist[j]=np.dot(input_data[:,i]-input_data[:,k_means[j]],input_data[:,i]-input_data[:,k_means[j]])#computes the distance of each data point from all means

        z[i]=np.argmin(dist)#assigns each data point to the cluster with the closest mean

    return z




df=pd.read_csv("A2Q1.csv", sep=',',header=None) #the data is read(please change to location of the file to match in the local computer)
dataset = df.to_numpy()
X=np.transpose(dataset)


z= kmeans(X,4)#we get the initialization using k means for 4 clusters
clust,m=Lloyds(X,z,4,1 )#we perform Lloyds for 4 clusters and for data with dimension 1


