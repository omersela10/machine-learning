from distutils.log import error
import math
import random
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt

def createXB (row, col):
    b=np.random.rand(col,1)
    for i in range(len(b)): # b=[1,2,3,4,...m]
        b[i][0]=i+1
    return np.random.rand(row,col),b

def createE(sigma,row):#e=[] normal(E[e]=0,sigma)
    e=np.random.rand(row,1)
    s = np.random.normal(0, int(sigma), row)
    for i in range(len(s)):
        e[i][0]=s[i]
    return e  
""" !!!!!!!!!!!!!!main!!!!!!!!!!!!!! """
#1.create x matrix with n exemple and m feature 
#2.create B and e and y=xb+e
#3. cheack the error on change sigma of e
#4.plot error as function of sigma
def main():
    m=100
    n=1500
    sigma=0
    x,b=createXB(n,m)
    
    xb = np.dot(x,b)

    z=np.linalg.inv(np.dot(x.T,x))
    z=np.dot(z,x.T)

    error = []
    sigma = [i for i in range(100)]
    #sigma change 0,1,...,99
    for i in range(100):
        e=createE(i,n)
        y=xb+e
        beta=np.dot(z,y)
        #beta= estimated b
        error.append((np.dot(abs((b-beta).T),abs(b-beta))[0][0]))
    

    print("B:")
    print (b)
    print("estimated B")
    print(beta)
    #plot error as function of sigma:
    plt.plot(sigma,error,'r.')
    plt.show()
    

   
if __name__ == "__main__":
    main()