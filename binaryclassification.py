from distutils.log import error
import math
import random
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from pathlib import Path
import pandas as pd

def sig(x):
    return 1/(1 + np.exp(-x))

def exelToMatrix(path1):
    return pd.read_excel(path1).to_numpy()

def binClassifier(test1, test2,train1,train2):
    N = len(train1) + len(train2)#2n
    x = np.concatenate((train1, train2), axis=0)#2nXd
    Wi = np.zeros((1,len(x[0])))  #1Xd
    for i in range(len(Wi)):
        Wi[i] = 0.001
    y=np.zeros((len(train1) + len(train2), 1))
    y[len(train1):] = 1

    epsilon = 0.000001
    epsilondivN = 0.001/N
    run = 100
    L = []
    iter = []

    for i in range(run):

        iter.append(i)
        der = np.zeros((1,len(x[0]))) #1Xd
        Xt = x  # 2nXd
        WiXt = np.dot(Xt, Wi.T)#nXdXdX1 = #2nX1
        sigm1 = sig(WiXt) #2nX1
        dif = y - sigm1 
        dif = np.dot(dif.T, Xt) #1X2nX2nXd = 1Xd
        Wi=Wi+epsilondivN*dif
        temp = np.sum(y*np.log(sigm1+epsilon))
        temp += np.sum((1-y)*np.log(1-sigm1+epsilon))

        L.append(temp/N)
        if(len(L)>=2)and(abs(L[-1])>=abs(L[-2])):
            L.pop()
            iter.pop()
            Wi=Wi-epsilondivN*dif
            return L , iter , Wi

    return L , iter , Wi

def successRateFunc(test1,test2, Wi):

    #test for one
    Y=np.dot(test1, Wi.T)#nxdXdx1=nx1
    successRate=0.0
    for i in range(len(Y)):
        Y[i][0]=sig(Y[i][0])
        if (Y[i][0])<=0.5:
            successRate+=1

    #test for two
    Y=np.dot(test2, Wi.T)#nxdXdx1=nx1
    for i in range(len(Y)):
        Y[i][0]=sig(Y[i][0])
        if (Y[i][0])>=0.5:
            successRate+=1
    return (successRate/(len(test1)+len(test2)))*100

""" !!!!!!!!!!!!!!main!!!!!!!!!!!!!! """

def main():

    print ("omer sela")
    #download the trains and tests files
    test1 = exelToMatrix("test1.xlsx")
    test2 = exelToMatrix("test2.xlsx")
    train1 = exelToMatrix("train1.xlsx")
    train2 = exelToMatrix("train2.xlsx")
    print(".....start:.......")

    #classifier func image "1" classefier 0 and imge "2" classefier 1.
    L , iter , Wi = binClassifier(test1, test2,train1,train2)

    print("Wi:", Wi)
    plt.plot(iter, L)
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.title('Loss function as a function of the number of iterations')
    plt.show()

    #success rate: test1+test2 with the best Wi
    print(" success Rate:", successRateFunc(test1,test2, Wi),"%")
   
if __name__ == "__main__":
    main()