import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,voronoi_plot_2d

#All the functions below are the same as the previous question
def means(input_data,clusters,no_cluster,d): # This function takes the input data and the clusters(along with number of clusters and dimension of data points d) and computes the means of each cluster
    mean_cluster=np.zeros([d,no_cluster ])
    count=np.zeros(no_cluster)
    for i in range(1000):
        mean_cluster[:,clusters[i]]+=input_data[:,i]
        count[clusters[i]]+=1.0
    for i in range(no_cluster):
        mean_cluster[:, i] = mean_cluster[:, i] * (1 / count[i])

    return mean_cluster



def Lloyds(input_data,initial_cluster,no_cluster,d):

    z_current=initial_cluster #z_current[i] contains the cluster to which the ith data point belongs to in the current iteration
    z_next=z_current #z_next contains the cluster numbers for data points after current iteration


    n_iteration=0
    while(1):

        z_current=z_next
        z_next=np.zeros(1000,dtype=int)

        mean=means(input_data,z_current,no_cluster,d)

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

            return z_next,mean


#This is the only new function from the previous question to this
#This question requires one to fix initialization for all k (2,3,4,5). Hence we first start from k=5 and with 5 random means;
#For the subsequent k(4,3,2) we use the first k of the initially selected 5 random means
# This function takes the index of the first 5 means,number of clusters(say b) and input data.
#It outputs the initialization of the dataset for b clusters considering the first b means
def initial(input_data,no_clusters,k_means):
    z=np.zeros(1000,dtype=int)
    for i in range(1000):
        dist=np.zeros(no_clusters)
        for j in range(no_clusters):
            dist[j]=np.dot(input_data[:,i]-input_data[:,k_means[j]],input_data[:,i]-input_data[:,k_means[j]])#computes the distance of each data point from all means

        z[i]=np.argmin(dist)#assigns each data point to the cluster with the closest mean

    return z

df=pd.read_csv('D:\Acad\PRML\Assignments\Assignment1\Dataset.csv', sep=',',header=None)
dataset = df.to_numpy()
X=np.transpose(dataset)

k_means=np.random.choice(1000,5,replace=False) #The first 5 means are initialized and fixed

no_clusters=5
while(no_clusters>=2):
    z=initial(X,no_clusters,k_means)
    clust,m=Lloyds(X,z,no_clusters,2)
    for i in range(1000): #plotting the data points
        if clust[i]==0:
            plt.plot(X[0,i],X[1,i],'ro') #red for cluster 1
        elif clust[i]==1:
            plt.plot(X[0,i],X[1,i],'bo') #blue for cluster 2
        elif clust[i]==2:
            plt.plot(X[0,i],X[1,i],'go') #green for cluster 3
        elif clust[i]==3:
            plt.plot(X[0,i],X[1,i],'yo') #yellow for cluster 4
        elif clust[i]==4:
            plt.plot(X[0,i],X[1,i],'co')

    v=np.zeros([no_clusters,2])
    for i in range(no_clusters):
        v[i,:]=m[:,i]


    plt.title('Plot of the data points with number of clusters='+str(no_clusters))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    if no_clusters!=2:# We use the voronoi library to plot the voronoi regions
        vorn=Voronoi(v)
        fig=voronoi_plot_2d(vorn,show_vertices=False,line_width=1.5,point_size=10)
    else: #This library works only for 3 or more means, hence we define seperate function for 2 means
        x_plot=np.arange(-10.0,10.0,0.1)
        midpt=(v[0,:]+v[1,:])*0.5
        per_direction=v[0,:]-v[1,:]
        per_slope=per_direction[1]/per_direction[0]
        slope=-1.0/per_slope
        y_plot=((x_plot-midpt[0])*slope)+midpt[1]
        plt.plot(x_plot,y_plot,'k',linestyle='dashed')
        plt.plot(v[0,0],v[0,1],'bo')
        plt.plot(v[1,0],v[1,1],'bo')
        plt.xlim([-10.0,10.0])
        plt.ylim([-10.0,10.0])



    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Plot of vornoi regions with k='+str(no_clusters))
    plt.show()
    no_clusters=no_clusters-1



