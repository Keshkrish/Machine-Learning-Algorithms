import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap


epsilon=0.25#This the tolerance
df=pd.read_csv("D:\Acad\PRML\Assignments\Assignment2\A2Q1.csv", sep=',',header=None)
dataset = df.to_numpy()
x=dataset
n=len(x)
K=4



def P_exp(x_i,beta_): #A function to compute the probability of a data point from an exponential function
    return (1.0/beta_)*(math.exp(-(1.0/beta_)*x_i))
def log_likelehood(beta,pi):#A function that computes the log of the likelehood. The input parameters are the model parameters(betas and pis)
    log_l=0.0
    for i in range(n):
        sum=0.0
        for k in range(K):

            sum+=P_exp(x[i],beta[k])*pi[k]


        log_l+=math.log(sum)

    return log_l




def EM():#The function that executes the Expectation maximisation algorithm

    beta_old=np.random.rand(K)*99 #The initial parameters of the model
    pi_old=np.random.rand(K)
    pi_old=pi_old/(np.sum(pi_old))#To ensure that the values of pi sum to 1

    lam=np.zeros((n,K)) #a matrix that stores the values of lamda
    change=1.0 #change is the variable that calculates the norm((theta)t-(theta)t+1). If this value is less than the tolerance epsilon then the algorithm is considered to be converged
    #It is assigned a value 1.0 initially so that the first iteration happens by default
    Log_L_array=[log_likelehood(beta_old,pi_old)]# This is a list that stores the log likelehood values fo every iteration
    while change>epsilon:


        #expectation step
        for i in range(n):
            for k in range(K):

                 lam[i,k]=pi_old[k]*P_exp(x[i],beta_old[k])#Lambda values are updated



            lam[i,:]=lam[i,:]/(np.sum(lam[i,:]))


        #maximization step


        beta=np.zeros(K)
        pi=np.zeros(K)
        for k in range(K):
            sum=0.0#a variable that stores the sum of the kth column in the lamda matrix
            beta[k]=0.0



            for i in range(n):
                sum+=lam[i,k]

            pi[k]=sum/n#pi is updated



            for i in range(n):
                beta[k]+=(lam[i,k]*x[i])/sum#beta is updated




        change=(((np.linalg.norm(pi-pi_old))**2)+((np.linalg.norm(beta-beta_old))**2))**0.5#norm((theta)t-(theta)t+1)=((norm((beta)t-(beta)t+1)^2+norm((pi)t-(pi)t+1)^2)^0.5

        Log_L_array.append(log_likelehood(beta,pi))#The log_likelehood after the current iteration is stored


        beta_old = beta#The beta and pi are updated for the next iteration

        pi_old = pi

    return Log_L_array

log_array_all=EM()#log_array_all is the array that will finally store the average of all the log_likelehood arrays
for i in range(1,100):
    log_array=EM()#this array stores the log_likelehood for the current random initialization
    # This if else is there so that if one of the arrays that have to be added has more elements than the other then the array with less elements is appended with its last element till the sizes of the 2 arrays are equal
    if len(log_array_all)>len(log_array):
        for i in range(len(log_array_all)-len(log_array)):
            log_array.append(log_array[len(log_array)-1])
    elif len(log_array)>len(log_array_all):
        for i in range(len(log_array)-len(log_array_all)):
            log_array_all.append(log_array_all[len(log_array_all)-1])

    for j in range(len(log_array_all)):#log_array_all at the present stores the sum of the log likelehood of all initializations
        log_array_all[j]+=log_array[j]

for i in range(len(log_array_all)):
    log_array_all[i]=log_array_all[i]/100.0#log_array_all at the present is normalised to get the average likelehood at each iteration

print('The average log likelihood after '+str(len(log_array_all))+' iterations is '+str(log_array_all[len(log_array_all)-1]))
plt.plot(range(len(log_array_all)),log_array_all)#Averages are plotted
plt.title("\n".join(wrap('Log likelihood for exponential mixture model(averaged over 100 random initializations) vs iteration number',60)))
plt.xlabel('iteration number')
plt.ylabel('Log Likelihood')
plt.show()
