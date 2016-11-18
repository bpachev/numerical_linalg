import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import diags

def arnoldi(Amul,b, nits=30):
    """
    Amul is a function that computes A
    """
    m = len(b)
    Q = np.zeros((m,nits+1))
    Q[:,0] = b/la.norm(b)
    H = np.zeros((nits+1, nits))
    for n in xrange(nits):
        v = Amul(Q[:,n])
        for j in xrange(n+1):
            H[j,n] = np.inner(Q[:,j], v)
            v -= np.inner(Q[:,j], v) * Q[:,j]
        
        H[n+1, n] = la.norm(v)
        Q[:, n+1] = v/H[n+1, n]
    return Q, H          

def ritz_values(Amul, b, nits):
    """
    Return all computed ritz values as a list of arrays, where the i-th array contains the ritz values at the i-th step
    """
    Q, H = arnoldi(Amul, b, nits)
    rvals = []
    for n in xrange(1,nits+1):
        rvals.append(la.eigvals(H[:n,:n]))
    return rvals

def scatter(arr,marker='o', s=20):
    plt.scatter(np.real(arr), np.imag(arr), marker=marker, s=s)

#problem 34(b)
def do_34b(N=64):
    ks = np.arange(1,N+1)**-.5
    A = diags([ks, ks[:-1]], offsets=[0,1])
    b = np.random.random(N)
    amul = lambda x: A.dot(x)
    rvals = ritz_values(amul, b, 30)
    evals = la.eigvals(A.todense())
    step=4
    for i in xrange(1,len(rvals),step):
        scatter(evals)
        scatter(rvals[i], marker="+", s=40)
        plt.title("34(b) Iteration {}".format(i+1))
        plt.show()

if __name__ == "__main__":
    do_34b()
