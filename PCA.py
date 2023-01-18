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


def matlabToPython(path):
    data=sio.loadmat(path)
    train = []
    lableTrain = []
    test = []
    lableTest = []
    i=0
    while (i < len(data['faces'])):
        for j in range(8):
            train.append(data['faces'][i+j])
            lableTrain.append(data['labeles'][i+j][0])
        i+=8
        for j in range(3):
            test.append(data['faces'][i+j])
            lableTest.append(data['labeles'][i+j][0])
        i+=3
    return np.array(train), np.array(lableTrain), np.array(test), np.array(lableTest)

def PCA(w ,v,k,evg,train):
    if(k>len(train[0]) or k<=0):
        return train

    wv=[[w[i],v[i]]for i in range(len(v))]
    wv.sort(key=lambda x: x[0],reverse=True)
    V=np.array([wv[i][1] for i in range(k)])
    for x in train:
        np.subtract(x,evg)
    Newtrain=np.dot(train, V.T)
    return Newtrain,V

def distance(x,y):
    return np.linalg.norm(x-y)

def KNN(train,lableTrain,test,lableTest):
    predictadeLableTest=[]
    for x in test:
        ans=[(distance(x,i)) for i in train]
        predictadeLableTest.append(lableTrain[np.argmin(ans)])
    return predictadeLableTest

    


def main():
    print ("PCA")
    path='C:\\Users\\omers\\source\\repos\\machine learning\\facesData.mat'
    train,lableTrain,test,lableTest = matlabToPython(path)
    evg=np.mean(train, axis=0)
    A =np.cov(train.T)
    w , v = np.linalg.eig(A)
    accurecy=[]

    maxK=1
    maxAccurecy=0


    for k in (tqdm(range(1,len(train[0])+1))):

        Newtrain,vec=PCA(w,v,k,evg,train)
        for x in test:
            np.subtract(x,evg)
        Newtest=np.dot(test,vec.T)
        predictadeLableTest=KNN(Newtrain,lableTrain,Newtest,lableTest)
        sum=0
        for i in range(len(predictadeLableTest)):
            if(predictadeLableTest[i]==lableTest[i]):
                sum+=1
        ac=sum/len(predictadeLableTest)*100
        accurecy.append(ac)
        if(ac>maxAccurecy):
            maxAccurecy=ac
            maxK=k
    plt.plot(accurecy)
    plt.show()
    print("max accurecy is: ",maxAccurecy," with k= ",maxK)

    
if __name__ == "__main__":
    main()