import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

"""
Benjamin Pachev
Math 510, Bigham Young University
Python code for 12.2 from Trefethen
"""


def vandermonde(x, deg=None):
    """
    Contruct the vandermonde matrix at the data points defined by x.
    If deg is provided, the matrix will be nxdeg
    INPUTS:
    x -- a 1-d numpy array

    Returns:
    V -- the associated vandermonde matrix
    """
    n = len(x)
    if deg is None:
        deg = n

    V = np.empty((n,deg))
    for i in xrange(deg):
        V[:,i] = x**i

    return V

def sup_norm(A):
    """
    Returns the infinity norm of A, a matrix or vector
    """
    return la.norm(A, ord=np.inf)

def mk_A(n,m):
    """
    Description:
        Construct an mxn matrix A that will take a vector (d_1, ..., d_n),
        find the interpolating polynomial at the n evenly spaced points x_1, ..., x_n on [0,1],
        and then evaluate the polynomial at each of the m evenly spaced y_1,...,y_m on [0,1].
    """
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,m)
    V_x = vandermonde(x, deg=n)
    V_y = vandermonde(y, deg=n)
    return V_y.dot(la.inv(V_x))

if __name__ == "__main__":
    dom = range(1,31)

    As = [mk_A(n, 2*n-1) for n in dom]
    #Part b
    norms = [sup_norm(A) for A in As]

    plt.plot(dom, np.array(norms))
    plt.yscale('log')
    plt.title("infinity norm of A with respect to n (part b)")
    plt.show()

    #part c
    #by 12.9 from trefethen, the condition number is norm(A) * norm(x)/norm(Ax)
    #pick x to be the all ones
    ks = []
    for i,A in enumerate(As):
        x = np.ones(i+1)
        ks.append(norms[i]*sup_norm(x)/sup_norm(A.dot(x)))

    plt.plot(dom, ks)
    plt.yscale('log')
    plt.title("infinity condition numbers of interpolating the constant function 1 (part c)")
    plt.show()

#    print ks[10]
    
