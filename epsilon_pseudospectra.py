import numpy as np
from scipy.sparse import diags
import scipy.linalg as la
from numpy.linalg import cond
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

def parta(A, n = 32):
    I = np.eye(n)
    def smallest_singular(z):
        return cond(A-z*I, p=-2)
    evals, evecs = la.eigh(A)
    epsilons = [10**(-k) for k in xrange(1,9)]
    gap = 1e-1
    ny, nx = 101, 100
    center = np.min(evals)
    x = np.linspace(-gap+center, gap+center, nx)
    y = np.linspace(-gap, gap, ny)
    X,Y = np.meshgrid(x, y)
    Z = np.empty_like(X)
    for i in xrange(ny):
        for j in xrange(nx):
            Z[i,j] = smallest_singular(x[j] + y[i] * 1j)
 #   m = np.min(evals)
 #   zs = 1j*np.linspace(0,gap, 100)
#    x = np.array([smallest_singular(m-z) for z in zs])
 #   plt.plot(zs,x)
    plt.figure()
    plt.axes().set_aspect('equal', 'datalim')
    CS = plt.contour(X,Y,Z, levels = epsilons[::-1])
    plt.scatter(np.sort(evals)[:3], np.zeros(3))
    plt.clabel(CS)
    plt.title("Contour plot of the spectrum near 3 eigenvalues.")
    plt.show()

def partb(A):
    evals, evecs = la.eigh(A)
    max_eval = np.max(evals)
    npoints = 100
    ts = np.linspace(0,50,npoints)
    y = np.array([cond(la.expm(t*A)) for t in ts])    
    plt.plot(ts, y)
    plt.plot(ts, np.exp(ts*max_eval))
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    n = 32
    A = diags([1,-1,1],[-1,0,1],shape=(n,n))
    A = A.todense()
#    parta(A)
    partb(A)

