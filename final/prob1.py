import numpy as np
from numpy.linalg import norm, lstsq
from numpy.random import random
import matplotlib.pyplot as plt
from scipy.sparse import diags
from functools import partial

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

def scale_csr_rows(B, d):
    """
    Scale the rows of a csr matrix
    """
    #The nonzero entries of B are stored in B.data
    #Because B is in csr format, B.data[inds[i]:inds[i+1]] contain the nonzero entries for the i-th row
    inds = B.indptr
    m = B.shape[0]
    for i in xrange(m):
        row_start_index = inds[i]
        row_stop_index = inds[i+1]
        #now scale the nonzeros entries in row i by the appropriate entry of d
        B.data[row_start_index:row_stop_index] *= d[i]            
    

def scale_csr_columns(B, d):
    """
    Scale each column of a csr matrix B, with the scalars for each column
    given in the array d
    """
    m = B.shape[0]
    inds = B.indptr
    for i in xrange(m):
        row_start_index = inds[i]
        row_stop_index = inds[i+1]
        col_arr = B.indices[row_start_index: row_stop_index]
        #The number of nonzeros in the current row
        #Equal to row_stop_index - row_start_index
        nonzero_columns = len(col_arr)
        for j in xrange(nonzero_columns):
            #Figure out the column number of this entry
            col_number = col_arr[j]
            #Use the column number to determine the appropriate scalar
            B.data[row_start_index+j] *= d[col_number]
            

def jacobi_precond(A, b, solver, **kwargs):
    """
    Compute a Jacobi preconditioner    
    INPUTS:
        A -- a sparse matrix
        b -- the right hand side
    
    RETURNS: (B,d)
    """
    d = A.diagonal()
    B = A.tocsr()
    scale_csr_rows(B, 1./d)

    #Solve the preconditioned system and then scale the solution appropriately
    guesses, res = solver(lambda x: B.dot(x), b/d, return_guesses=True, **kwargs)
    sol = guesses[-1]
    #We need to fix the residuals because the error from the preconditioned system won't be the same as the actual error
    for i, x in enumerate(guesses):
        res[i] = norm(A.dot(x)-b)
    return sol, res


def symmetric_jacobi(A, b, solver, **kwargs):
    """
    Like the previous preconditioner except it does right and left preconditioning.
    i.e D^(-1/2) A D^(-1/2) D^(1/2) x = D^(-1/2) b
    This is needed in order to preserve symmetry.     
    """
    d = A.diagonal()
    B = A.tocsr()
    #Compute the scaling array d^(-1/2)
    s = 1. / np.sqrt(d)
    #Compute B = D^(-1/2) A D^(-1/2)
    scale_csr_rows(B, s)
    scale_csr_columns(B, s)

    #Solve the preconditioned system and then scale the solution appropriately
    guesses, res = solver(lambda x: B.dot(x), b*s, return_guesses=True, **kwargs)
    sol = guesses[-1]*s
    #We need to fix the residuals because the error from the preconditioned system won't be the same as the actual error
    for i, x in enumerate(guesses):
        res[i] = norm(A.dot(s*x)-b)
    return sol, res
    

def mk_A(n=200):
    if n > 100:
        ds = [np.ones(n-100), np.ones(n-1), np.sqrt(np.arange(1,n+1)) + .5, np.ones(n-1), np.ones(n-100)]
        A = diags(ds, offsets=[-100,-1,0,1,100])
    else:
        ds = [np.ones(n-1), np.sqrt(np.arange(1,n+1)) + .5, np.ones(n-1)]
        A = diags(ds, offsets=[-1,0,1])    
    return A


"""
Here we put the arnoldi and GMRES code.
I coded up a GMRES algorithm with restarts for efficiency and for kicks.
"""

def arnoldi(Amul,b, nits=30):
    """
    Amul is a function that computes A
    """
    m = len(b)
    Q = np.zeros((m,nits+1))
    Q[:,0] = b/norm(b)
    H = np.zeros((nits+1, nits))
    for n in xrange(nits):
        v = Amul(Q[:,n])
        for j in xrange(n+1):
            H[j,n] = np.inner(Q[:,j], v)
            v -= np.inner(Q[:,j], v) * Q[:,j]
        
        H[n+1, n] = norm(v)
        Q[:, n+1] = v/H[n+1, n]
    return Q, H          

def gmres(Amul, b, nits=1, return_guesses=False, tol=1e-6):
    Q, H = arnoldi(Amul, b, nits)
    residuals = []
    normb = norm(b)
    if return_guesses: guesses = []
    for n in xrange(1, nits+1):
        e1 = np.zeros(n+1)
        e1[0] = normb
        y, res = lstsq(H[:n+1, :n], e1)[:2]
        residuals.append(res[0])
        if return_guesses: guesses.append(Q[:,:n].dot(y))
        if residuals[-1] < tol: break
    
    if return_guesses: return guesses, residuals    
    else: return Q[:,:n].dot(y), residuals

#IGNORE, not used
def restarted_gmres(Amul, b, max_its=50, tol=1e-6, step_size=10, return_guesses=False):
    """
    step_size is the number of dimensions to compute before restarting
    """
    if return_guesses: guesses = [b]
    residuals = []
    sol = b
    its_left = max_its
    while its_left > 0:
        its_this_step = min(step_size,  its_left)
        if return_guesses:
            step_guesses, step_residuals = gmres(Amul, sol, nits=its_this_step, tol=tol, return_guesses=True)
            guesses += step_guesses
            sol = guesses[-1]
        else:
            sol, step_residuals = gmres(Amul, sol, nits=its_this_step, tol=tol, return_guesses=False)
        residuals += step_residuals
        if residuals[-1] < tol: break
        max_its -= step_size
    
    if return_guesses:
        return guesses, residuals
    else:
        return sol, residuals

def plot_convergence(A, b, solver, solver_name):
    n = A.shape[0]
    amul = lambda x: A.dot(x)
    sol, res = symmetric_jacobi(A, b, solver, tol=0)    
    plt.plot(res, label="Symmetrically Preconditioned")
    rsol, rres = jacobi_precond(A, b, solver, tol=1e-14)
    plt.plot(rres, label="Left Preconditioned")
    psol, pres = solver(amul, b,  tol=1e-6)
    plt.plot(pres, label="Normal")
    plt.title("Preconditioned {} convergence on large, sparse {}x{} system".format(solver_name,n,n))
    plt.ylabel("Residual")
    plt.xlabel("Iterations")
    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ns = [100,200, 500, 1000, 10000]
    for n in ns:
        A = mk_A(n)
        b = np.ones(n)
        plot_convergence(A, b, partial(CG, max_its=40), "CG")
        plot_convergence(A,b,partial(gmres, nits=30), "gmres")
    
