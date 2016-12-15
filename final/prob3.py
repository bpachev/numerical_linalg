import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import eigvals_banded

def lanczos(amul, b, nits):
    m = len(b)
    Q = np.zeros((m,nits+1))
    beta = np.zeros(nits)
    alpha = np.zeros(nits)
    Q[:,0] = b/la.norm(b)
    for n in xrange(0, nits):
        v = amul(Q[:,n])
        alpha[n] = np.inner(Q[:,n], v)
        #Note that if n=0. beta[-1] = 0 (since we initialized it to 0), as it should be
        v -= beta[n-1] * Q[:,n-1] + alpha[n] * Q[:,n]
        beta[n] = la.norm(v)
        Q[:, n+1] = v/beta[n]
    return alpha, beta
    
def ritz_values(Amul, b, nits):
    """
    Return all computed ritz values as a list of arrays, where the i-th array contains the ritz values at the i-th step
    """
    alpha, beta = lanczos(Amul, b, nits)
    rvals = []
    
    #This puts the diagonals into the appropriate form for eigvals_banded
    M = np.zeros((2,nits))
    M[0,1:] = beta[:-1]
    M[1] = alpha
   
    for n in xrange(1,nits+1):
        rvals.append(eigvals_banded(M[:,:n]))
    return rvals


if __name__ == "__main__":
    n = 1000    
    nits = 40
    #A = np.random.random((n,n))
    #A = A.dot(A.T)
    #Construct the matrix
    ds = [np.ones(n-100), np.ones(n-1), np.sqrt(np.arange(1,n+1)), np.ones(n-1), np.ones(n-100)]
    A = diags(ds, offsets=[-100,-1,0,1,100])
    b = np.ones(n)
    Amul = lambda x: A.dot(x)
    rvals = ritz_values(Amul, b, nits)
    #Convert the matrix A into the wierd form required by eigvals_banded
    M = np.zeros((101,n))
    M[-1] = ds[2]
    M[-2,1:] = ds[1]
    M[0,100:] = ds[0]
    print "Eigenvalue of A from 36.3 obtained after {0} iterations: {1:.7f}".format(nits, rvals[-1][0])
    #This call will get only the smallest eigenvalue
    print "True smallest eigenvalue (according to scipy): {0:.7f}".format(eigvals_banded(M, select='i', select_range=(0,0))[0])
