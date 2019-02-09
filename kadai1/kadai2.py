from matplotlib import pyplot as plot
import numpy as np
from scipy.io import wavfile as wav
import sys

def getMat(filename1,filename2):
    data = open(filename1, "r")
    arr1 = []
    for val in data:
        arr1.append(float(val))
    data.close()
    data = open(filename2, "r")
    arr2 = []
    for val in data:
        arr2.append(float(val))
    data.close()
    mat = np.empty((2,len(arr1)))
    mat[0] = arr1; mat[1] = arr2
    return mat, 2, len(arr1)


def plotMat(mat,dim):
    arr = np.asarray(mat)
    plot.plot(arr[0])
    plot.plot(arr[1])
    plot.show()

def whitening(x,dim): 
    sigma = np.cov(x)
    _, E = np.linalg.eig(sigma)
    E_inv = np.linalg.inv(E)
    D = E_inv.dot(sigma).dot(E)
    D = np.multiply(D,np.eye(dim)) # 対角成分をのこす
    D = np.linalg.inv(np.sqrt(D)) # D^(-1/2)
    V = E.dot(D).dot(E_inv)
    return np.asmatrix(V.dot(x))

def normalization(w):
    if w.sum() < 0:
        w *= -1
    norm = np.linalg.norm(w)
    return w/norm

def orthogonalization(W,i,w,size):
    sum = [0]*dim
    sum = np.matrix([sum]).T
    for n in range(i):
        wn = W[n].T
        sum = sum + (wn.T*w)[0,0] * w
    return w - sum


def optimisation(z,dim,size):
    W = np.empty((dim,dim))
    W = np.matrix(W)
    Y = np.empty((dim,size))
    Y = np.matrix(Y)
    print("")
    for n in range(dim):

        w = np.random.rand(dim,1)
        w = np.asmatrix(w)    
        w = normalization(w)
        while True:
            prew = w
            k = np.asmatrix((np.asarray(z) * np.asarray(w.T * z) ** 3).mean(axis=1)).T
            w = k - 3*w
            w = orthogonalization(W,n,w,size)
            w = normalization(w)
            norm = np.linalg.norm(w - prew)
            if norm < 0.000001:
                W[n] = w.T
                break
    return W

if __name__ == "__main__":
    np.random.seed(20190205)

    path1 = "1_1/dat1.txt"
    path2 = "1_1/dat2.txt"
    x, dim, size = getMat(path1,path2)

    z = whitening(x,dim)
    # plotMat(z,dim)

    print(np.cov(z,rowvar=False))

    W = optimisation(z,dim,size)

    print("")

    y = W * z

    plotMat(y,dim)