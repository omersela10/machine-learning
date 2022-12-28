from distutils.log import error
import math
import random
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from pathlib import Path
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
#from sklearn.preprocessing import PolynomialFeatures


def matlabToPython(path):
    data=sio.loadmat(path)
    train = []
    test = []
    for i in data.keys():
        if i.startswith('train'):
            train.append(data[i])
        elif i.startswith('test'):
            test.append(data[i])
    featureNum = len(train[0])
    return train, test , featureNum

def evg(M,lenfeature):
    sum = [0 for i in range(lenfeature)]
    if len(M)>0:
        for i in range(len(M)):
            for j in range(len(M[i])):
                sum[j] += M[i][j]
        for i in range(len(sum)):
            sum[i] /= len(M)
    return sum

def succesRate(c,test):#len(c)=k,len(test)=k,
    print("succesRate")
    count = 0
    sum = 0
    for i in range(len(test)):
        for j in range(len(test[i])):
            sum+=1
            dist = [0 for t in range(len(c))]
            for q in range(len(c)):
                dist[q] = np.linalg.norm(test[i][j]-c[q])
            if dist.index(min(dist)) == i:
                count += 1
    return 100*count/sum
def LOSS(M,c):
    loss = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            loss += np.linalg.norm(M[i][j]-c[i])
    return loss

def randomdots(data,k):
    maxVec = [data[0][i] for i in range(len(data[0]))]
    minVec = [data[0][i]for i in range(len(data[0]))]
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > maxVec[j]:
                maxVec[j] = data[i][j]
            if data[i][j] < minVec[j]:
                minVec[j] = data[i][j]
    c = [[0 for i in range(len(data[0]))] for j in range(k)]
    step = [(maxVec[i]-minVec[i])/k for i in range(len(data[0]))]
    for i in range(k):
        for j in range(len(data[0])):
            c[i][j] = (i+1)*step[j]
    return c

""" !!!!!!!!!!!!!!main!!!!!!!!!!!!!! """
def main():

    print("omer sela")
    train, test , feature= matlabToPython('mnist_all.mat')
    k=10
    data = []
    
    for i in train:
        for j in i:
            data.append(j)
    print(len(data),len(data[0]),len(data[-1]))

    c = [[train[i][0]] for i in range(k)]
    #c = randomdots(data,k)
    for i in c:
        print(len(i))
    #data[random.randint(0,len(data)-1)
    i=0
    los1=0
    los2=0
    loss=[]
    c1=c
    c2=[]
    while True:
        M= [[] for i in range(k)]
        for j in (tqdm(range(len(data)))):
            dist = [0 for q in range(len(c))]
            for q in range(len(c)):
                dist[q] = np.linalg.norm(data[j]-c[q])
            M[dist.index(min(dist))].append(data[j])
        for j in range(len(c)):
            c[j]=evg(M[j],len(data[0]))
        #cheack loss:
        i+=1
        if i==1:
            los1=LOSS(M,c)
            c1=c
            loss.append(los1)
        else:
            los2=LOSS(M,c)
            c2=c
            loss.append(los2)
            if  (i>6 and (los1<=los2 or min(los1,los2)/max(los1,los2)>=0.99))or i>10:
                c2=c1
                break
            los1=los2
            c1=c2
    c=c1
    plt.plot(loss,".r")
    plt.show()
    print("succesRate1")
    print(c)
    print("the succes rate is:", succesRate(c,test))

   
    #train is a arry of trains: train[0]= exemples of the numbers '0'
    #train[0][0] = the first example of the number '0'
    #train[0][0][0] = the first pixel/feature of the first example of the number '0'



    
if __name__ == "__main__":
    main()
