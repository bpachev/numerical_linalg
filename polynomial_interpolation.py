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
    print V, n, x

    return V

def mk_A(n,m):
    """
    Description:
        Construct an mxn matrix A that will take a vector (d_1, ..., d_n),
        find the interpolating polynomial at the n evenly spaced points x_1, ..., x_n on [0,1],
        and then evaluate the polynomial at each of the m evenly spaced y_1,...,y_m on [0,1].
    """
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    V_x = vandermonde(x, deg=n)
    V_y = vandermonde(y, deg=n)
    return V_y.dot(la.inv(V_x))

if __name__ == "__main__":
    #Part b
    norms = []
    for n in xrange(1,31):
        m = 2*n-1
        norms.append(la.norm(mk_A(n,m), ord=np.inf))

    plt.plot(range(1,31), np.array(norms))
    plt.yscale('log')
    plt.show()
