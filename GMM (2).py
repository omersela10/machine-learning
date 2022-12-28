from distutils.log import error
import math
import random
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
from scipy.stats import multivariate_normal

#omer sela
#Gaussian mixture model of clustering

#generate expectation:
def expactation(k):
    rnd = random.random()*10
    exp = np.zeros((k,2))
    for i in range(k):
        exp[i] = (10*i + rnd ,10*i + rnd)
    return exp
#generate covariance matrices:
def variance(k):
    var = [np.zeros((2,2)) for i in range(k)]
    factor = 5
    for i in range(k):
        var[i][0][0] = factor*random.random()
        var[i][1][1] = factor*random.random()
    return var
#generate distribuition of the data
def generate_Alpha(k):
    alpha = np.zeros((k,1))
    for i in range(k):
        alpha[i] = random.random()
    sum = 0
    for i in range(k):
        sum += alpha[i]
    for i in range(k):
        alpha[i] /= sum
    return alpha
#generate data points
def generateData(k,exp,var,alpha,m):

    data = [[] for i in range(k)]
    temp = [int(alpha[i]*m) for i in range(k)]
    temp[-1] = m - sum(temp[:-1])
    ans = np.zeros((m,2))
    c = 0
    for i in range(k):
        data[i] = (np.random.multivariate_normal(exp[i],var[i],temp[i]))
    for i in range(k):
        for j in range(len(data[i])):
            ans[c][0] = data[i][j][0]
            ans[c][1] = data[i][j][1]
            c += 1
    return ans ,temp

# calculate success rate
def successRateGMM(data,exp,var,alpha,k,numSamplesData):

    w = [[0 for j in range(len(data))] for i in range(k)]
    for i in range(k):
        for j in range(len(data)):
            mn = multivariate_normal(exp[i],var[i])
            w[i][j] = alpha[i]*mn.pdf(data[j])
    label = []
    for j in range(len(data)):
        temp = [w[i][j] for i in range(k)]
        label.append(temp.index(max(temp)))
    
    trueLabel=[]
    for i in range(k):
        temp = [i for j in range(numSamplesData[i])]
        for j in temp:
            trueLabel.append(j)
    count = 0
    for i in range(len(label)):
        if label[i] == trueLabel[i]:
            count+=1
    return 100*count/len(label)

def GaussianMixtureModel(data,k, iter):

    reg_cov = 1e-6*np.identity(len(data[0]))
    alpha = np.ones(k)/k
    xMin = yMin = math.inf
    xMax = yMax = -1
    #choose wisely k points of initial expectations
    for i in range(len(data)):
        xMin = min(xMin,data[i][0])
        yMin = min(yMin,data[i][1])
        xMax = max(xMax,data[i][0])
        yMax = max(yMax,data[i][1])
    step = [((xMax-xMin)/(k+1)),((yMax-yMin)/(k+1))]

    exp = np.zeros((k,len(data[0])))
    for i in range(k):
        exp[i] = ((i+1)*step[0],(i+1)*step[1])

    # assume first covariance matrix is the unit matrix times 5
    var = np.zeros((k,len(data[0]),len(data[0]))) 
    for dim in range(len(var)):
        np.fill_diagonal(var[dim],5)


    log_likelihoods = []
    mu = []
    cov = []
    R = []
    
    # GMM algortihm:
    for i in range(iter):
        mu.append(exp)
        cov.append(var)
        r_ic = np.zeros((len(data),len(var)))
        #E step:
        for m,co,a,r in zip(exp,var,alpha,range(len(r_ic[0]))):
            co += reg_cov
            mn = multivariate_normal(mean = m,cov = co)
            r_ic[:,r] = a * mn.pdf(data)/np.sum([pi_c*multivariate_normal(mean = mu_c,cov = cov_c).pdf(data) for pi_c,mu_c,cov_c in zip(alpha,exp,var+reg_cov)],axis=0)
        R.append(r_ic)
         # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
        exp = []
        var = []
        alpha = []
        log_likelihood = []
        # M step
        for c in range(len(r_ic[0])):
            m_c = np.sum(r_ic[:,c],axis = 0)
            mu_c = (1/m_c)*np.sum(data*r_ic[:,c].reshape(len(data),1),axis = 0)
            exp.append(mu_c)

            # Calculate the covariance matrix per source based on the new mean
            var.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(data),1)*(data-mu_c)).T,(data-mu_c)))+ reg_cov)
            # Calculate new alpha which is the "fraction of points" respectively the fraction of the probability assigned to each source 
            alpha.append(m_c/np.sum(r_ic)) 
   
    # Log likelihood
        log_likelihoods.append(np.log(np.sum([k*multivariate_normal(exp[i],var[j]).pdf(data) for k,i,j in zip(alpha,range(len(exp)),range(len(var)))])))

    
  
    return alpha,exp,var,log_likelihoods
    
            



def main():

    print ("gaussianMixture")
    k = 3
    alpha = generate_Alpha(k)
    exp = expactation(k)
    var = variance(k)
    m = 1000
    iter = 100
    data , numSamplesData = generateData(k,exp,var,alpha,m)
    x = [data[i][0] for i in range(m)]
    y = [data[i][1] for i in range(m)]
    plt.plot(x,y,".r")
    plt.show()
    
    alpha,exp,var,log_likelihood= GaussianMixtureModel(data,k,iter)
    print(exp)
    print(var)
    N = [i for i in range(iter)]
    plt.plot(N, log_likelihood)
    plt.show()
    sr = successRateGMM(data,exp,var,alpha,k,numSamplesData)
    print("success rate is :", sr ,"%")

if __name__ == "__main__":
    main()