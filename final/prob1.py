import numpy as np
from scipy.linalg import norm
from numpy.random import random
import matplotlib.pyplot as plt
from scipy.sparse import diags

def CG(Amul, b, tol=1e-6, max_its = 100):
    """
    A simple conjugate gradient solver for solving a positive definite linear system A*x = b.
    Amul -- a function for computing A*x for arbitrary x
    b -- the initial guess
    tol -- When residuals fall below tol, the algorithm terminates
    max_its -- the maximum number of iterations
    Returns: (x, residuals)
    x -- the solution
    residuals -- a list of residual norms (i.e to plot with)
    """
    #Initialization
    residuals = []
    m = len(b)
    b = b.astype(np.float64)
    #Initial x
    x = np.zeros(m)
    #Initial residual vector
    r = np.copy(b)
    residuals.append(norm(r))
    #Initial search direction
    p = np.copy(r)
    for i in xrange(max_its):
        #check for convergence
        if residuals[i] < tol: break

        #The conjugate gradient algorithm as given in Trefethen page 294
        #No need to compute this twice
        Ap = Amul(p)
        #We are using the two-norm, so r^T *r = ||r||^2
        old_r_squared = np.inner(r,r)
        alpha = old_r_squared / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        residuals.append(norm(r))
        #Dividing before squaring should help with stability
        beta = np.inner(r,r)/old_r_squared
        p = r + beta * p

    return x, residuals

def mk_A(n=100):
    ds = [np.ones(n-100), np.ones(n-1), np.sqrt(np.arange(1,n+1)) + .5, np.ones(n-1), np.ones(n-100)]
    A = diags(ds, offsets=[-100,-1,0,1,100])
    return A

if __name__ == "__main__":
    n = 1000
    A = mk_A(n)
    amul = lambda x: A.dot(x)
    b = random(n)
    sol, res = CG(amul, b, max_its=50)
    plt.plot(res)
    plt.title("CG convergence on random {}x{} system".format(n,n))
    plt.ylabel("Residual")
    plt.xlabel("Iterations")
    plt.yscale("log")
    plt.show()
