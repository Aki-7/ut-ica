from matplotlib import pyplot as plot
import numpy as np
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
    return np.asmatrix(V.dot(x)).T

def normalization(w):
    if(w.sum() < 0):
        w *= -1
    norm = np.linalg.norm(w)
    return w/norm

def orthogonalization(W,i,w,size):
    sum = [0]*size
    sum = np.matrix([sum]).T
    for n in range(i):
        wn = np.matrix([W[n]]).T
        sum = (wn.T*w)[0,0] * wn
        # sum = sum + (w * wn.T) * wn
        # sum = sum + (wn * w.T) * wn
    return w - sum


def optimisation(z,dim,size):
    W = np.empty((size,size))
    m1 = -1; m2 = -2
    print("")
    for n in range(size):

        m2 = (size-n)*150/size
        if(m1 is not m2):
            sys.stdout.write("\033[2K\033[G")
            print("[",end="")
            for i in range(150):
                if(i < 150-m1):
                    print("-",end="")
                else:
                    print(" ",end="")
            print("]",end="")
            sys.stdout.flush()
            m1 = m2

        w = np.random.rand(size,1)
        w = normalization(w)
        while True:
            prew = w
            w = np.asmatrix((np.asarray(z) * np.asarray(w.T * z) ** 3).mean(axis=1)).T - 3*w
            w = orthogonalization(W,n,w,size)
            w = normalization(w)
            norm = np.linalg.norm(w - prew)
            # print(norm)
            if norm < 0.1:
                W[n] = np.asarray(w.T)
                break
    return np.asmatrix(W)

if __name__ == "__main__":
    # np.random.seed(123)

    path1 = "1_1/dat1.txt"
    path2 = "1_1/dat2.txt"
    x, dim, size = getMat(path1,path2)

    z = whitening(x,dim)
    plotMat(z.T,dim)

    print(np.cov(z,rowvar=False))
    print(z)

    W = optimisation(z,dim,size)

    y = W.T * z

    plotMat(y.T,dim)