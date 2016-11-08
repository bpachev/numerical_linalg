from numpy import array, eye, dot, zeros
import numpy as np
from scipy.linalg import eig
from qr_algorithm import hilb

def ldl(A):
    """
    Some code I stole from the source code of Numpy
    """
    n = A.shape[1]
    L = array(eye(n))
    D = zeros((n, 1))
    for i in range(n):
        D[i] = A[i, i] - dot(L[i, 0:i] ** 2, D[0:i])
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - dot(L[j, 0:i] * L[i, 0:i], D[0:i])) / D[i]
    D = array(eye(n)) * D
    return [L, D, L.T]

def get_shift(a,b):
    if a == -np.inf:
        if b == np.inf:
            return 0
        else:
            return -2*(abs(b) + 1)
    else:
        if b == np.inf:
            return 2*(abs(a) + 1)
        else: return (a+b)/2.
       

def nu(A, shift, tol=1e-15):
    """
    Determine how many eigenvalue of A are < shift and > shift
    """
    n = A.shape[0]
    if shift == np.inf:
        return 0, n
    elif shift == -np.inf:
        return n, 0
    D = ldl(A-shift*np.eye(n))[1]
    vals = np.diag(D).copy()
    vals[np.abs(vals) < 1e-15] = 0
    npos = np.sum(vals > 0)
    nneg = np.sum(vals < 0)
    return npos, nneg

def bisection(A, a=-np.inf, b = np.inf, nvals = None, tol=1e-7):
    """
    Find all eigenvalues of A using the bisection algorithm with ldl decomposition
    This is a recursive function
    nvals is the number of eigenvalues currently contained in the given interval
    """
    n = A.shape[0]
    if nvals is None:
        nvals = A.shape[0]
    elif nvals == 0: return []
    elif abs(b-a) < tol:
        return nvals * [(a+b)/2.]
#    print a,b
    shift = get_shift(a,b)
    
    npos, nneg = nu(A, shift)
    lpos, lneg = nu(A, a)
    upos, uneg = nu(A, b)
    nzero = n-npos-nneg
    
    return bisection(A, a,shift, nvals=nneg-lneg-nzero) + (nzero)*[shift] + bisection(A, shift, b, nvals=uneg - nneg - nzero)
    
    
            
    
if __name__ == "__main__":
    A = hilb(4) + np.pi * np.eye(4)
    A = np.random.random((4,4))
    A += A.T
    print "A", A
    print "Real eigenvalues ", np.sort(eig(A)[0])
    print "computed by bisection ", np.sort(bisection(A, tol=1e-10))

