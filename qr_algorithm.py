import numpy as np
from householder import house, formQ, sign, ej
from scipy.linalg import norm

def hilb(n):
    a = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            a[i,j] = 1./(i+j+1)
    return a

def tridiag(A):
    m = A.shape[0]
    R = A.copy()
    W = np.zeros(A.shape, dtype = R.dtype)
    for k in xrange(0,m-1):
        x = R[k+1:,k]
        dim = len(x)
        v = sign(x[0]) * norm(x) * ej(dim) + x
        v = v/norm(v)
        W[k+1:, k+1] = v
        #Multiply the appropriate submatrix of A by the reflector
        R[k+1:,k:] -= 2*np.dot(v.reshape((dim, 1)), np.dot(v.reshape((1,dim)).conj(), R[k+1:,k:]))
        R[:,k+1:] -= 2*np.outer(np.dot(R[:,k+1:], v), v.conj())

    return W, R


if __name__ == "__main__":
    A = hilb(4)
    W, T = tridiag(A)
