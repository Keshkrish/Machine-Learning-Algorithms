import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap


epsilon=0.25
df=pd.read_csv("A2Q1.csv", sep=',',header=None)
dataset = df.to_numpy()
x=dataset
n=len(x)
K=4


def P_gauss(x_i,nu_,sigma_):#A function to compute the probability of a data point from a guassian function

    p=math.exp(-0.5*(x_i-nu_)*(x_i-nu_)/(sigma_*sigma_))
    p=p/((2*3.14*sigma_*sigma_)**0.5)
    return p

def log_likelehood(nu,sigma,pi):#A function that computes the log of the likelehood. The input parameters are the model parameters(nus,sigmas and pis)
    log_l=0.0
    for i in range(n):
        sum=0.0
        for k in range(K):

            sum+=P_gauss(x[i],nu[k],sigma[k])*pi[k]


        log_l+=math.log(sum)

    return log_l




def EM():#a function to perform EM algorithm(similar to the previous question)
    nu_old=np.random.rand(K)*99
    sigma_old=np.random.rand(K)*100+1
    pi_old=np.random.rand(K)
    pi_old=pi_old/(np.sum(pi_old))

    lam=np.zeros((n,K))
    change=1.0
    Log_L_array=[log_likelehood(nu_old,sigma_old,pi_old)]
    while change>epsilon:


        #expectation step
        for i in range(n):
            for k in range(K):

                 lam[i,k]=pi_old[k]*P_gauss(x[i],nu_old[k],sigma_old[k])



            lam[i,:]=lam[i,:]/(np.sum(lam[i,:]))


        #maximization step

        nu=np.zeros(K)
        sigma=np.zeros(K)
        pi=np.zeros(K)
        for k in range(K):
            sum=0.0
            nu[k]=0.0
            sigma[k]=0.0


            for i in range(n):
                sum+=lam[i,k]

            pi[k]=sum/n



            for i in range(n):
                nu[k]+=(lam[i,k]*x[i])/sum
            for i in range(n):
                sigma[k]+=(lam[i,k]*((x[i]-nu[k])**2))/sum
            sigma[k]=(sigma[k])**0.5



        change=(((np.linalg.norm(pi-pi_old))**2)+((np.linalg.norm(nu-nu_old))**2)+((np.linalg.norm(sigma-sigma_old))**2))**0.5

        Log_L_array.append(log_likelehood(nu,sigma,pi))



        nu_old = nu
        sigma_old = sigma
        pi_old = pi

    return Log_L_array

log_array_all=EM()
for i in range(1,100):
    log_array=EM()
    if len(log_array_all)>len(log_array):
        for i in range(len(log_array_all)-len(log_array)):
            log_array.append(log_array[len(log_array)-1])
    elif len(log_array)>len(log_array_all):
        for i in range(len(log_array)-len(log_array_all)):
            log_array_all.append(log_array_all[len(log_array_all)-1])

    for j in range(len(log_array_all)):
        log_array_all[j]+=log_array[j]

for i in range(len(log_array_all)):
    log_array_all[i]=log_array_all[i]/100.0

print('The average log likelihood after '+str(len(log_array_all))+' iterations is '+str(log_array_all[len(log_array_all)-1]))
plt.plot(range(len(log_array_all)),log_array_all)
plt.title("\n".join(wrap('Log likelihood for guassian mixture model(averaged over 100 random initializations) vs iteration number',60)))
plt.xlabel('iteration number')
plt.ylabel('Log Likelihood')
plt.show()
