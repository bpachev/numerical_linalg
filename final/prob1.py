import numpy as np
from scipy.linalg import norm
from numpy.random import random
import matplotlib.pyplot as plt
from scipy.sparse import diags

def CG(Amul, b, tol=1e-6, max_its = 100, return_guesses=False):
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
    if return_guesses: guesses = [x]
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
        x = x + alpha * p
        r -= alpha * Ap
        residuals.append(norm(r))
        if return_guesses: guesses += [x]
        #Dividing before squaring should help with stability
        beta = np.inner(r,r)/old_r_squared
        p = r + beta * p

    if return_guesses: return guesses, residuals
    else: return x, residuals

def jacobi_precond(A, b, solver, **kwargs):
    """
    Compute a Jacobi preconditioner    
    INPUTS:
        A -- a sparse matrix
        b -- the right hand side
    
    RETURNS: (B,d)
        B -- D^(-1) * A, where D is the diagonal matrix whose diagonal is the same as A's
        d -- an array with the diagonal elements of A
        
    """
    d = A.diagonal()
    B = A.tocsr()
    #The nonzero entries of B are stored in B.data
    #Because B is in csr format, B.data[inds[i]:inds[i+1]] contain the nonzero entries for the i-th row
    inds = B.indptr
    m = B.shape[0]
#    print m, len(inds)
    for i in xrange(m):
        row_start_index = inds[i]
        row_stop_index = inds[i+1]
        #now scale the nonzeros entries in row i by the appropriate diagonal entry of A
        #We don't want any monkey business with division by zero
        if abs(d[i]) > 1e-15: B.data[row_start_index:row_stop_index] /= d[i]            
        #Don't scale that row
        else: d[i] = 1
    print B.data
    print A.tocsr().data
    #Solve the preconditioned system and then scale the solution appropriately
#    print d, b/d
    guesses, res = solver(lambda x: B.dot(x), b/d, return_guesses=True, **kwargs)
    sol = guesses[-1]*d
    #We need to fix the residuals because the error from the preconditioned system won't be the same as the actual error
    for i, x in enumerate(guesses):
        res[i] = norm(d*x-b)
    return sol, res

def mk_A(n=200):
    ds = [np.ones(n-100), np.ones(n-1), np.sqrt(np.arange(1,n+1)) + .5, np.ones(n-1), np.ones(n-100)]
    A = diags(ds, offsets=[-100,-1,0,1,100])
    return A

if __name__ == "__main__":
    n = 1000
    A = mk_A(n)
    b = np.ones(n)
    amul = lambda x: A.dot(x)
    sol, res = jacobi_precond(A, b, CG, max_its=50, tol=1e-8)    
    sol, res = CG(amul, b, max_its=50, tol=1e-8)
    plt.plot(res)
    plt.title("Preconditioned CG convergence on large, sparse {}x{} system".format(n,n))
    plt.ylabel("Residual")
    plt.xlabel("Iterations")
    plt.yscale("log")
    plt.show()
