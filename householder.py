import numpy as np
from numpy.linalg import norm

def ej(d,j=0):
    """
    Return the e_k-th basis vector of dimension d, starting at 0.
    """
    res = np.zeros(d)
    res[j] = 1
    return res

def sign(a):
    """
    Return the sign of a real number, or the complex phase for a complex number
    """
    if a == 0:
        return 1
    else: return a/abs(a)

def house(A):
    """
    Compute the implcit QR decomposition of a mxn complex numpy matrix A.
    Returns:
        W -- A lower-triangular complex mxn matrix, whose columns are the vs used for householder reflections.
        R -- The upper triangular nxn matrix in the QR decomposition.
    """
    m, n = A.shape
    if n > m: raise ValueError("The array must NOT have more columns than rows")

    R = A.astype(np.complex128)
    W = np.zeros(A.shape, dtype = R.dtype)

    for k in xrange(n):
        #select out the x-vector in the k-th column of R that needs to be reflected onto the span of e_1
        x = R[k:, k]
        dim = m-k
        #now pick an appropriate v for the Householder reflection
        v =  sign(x[0]) * norm(x) * ej(dim) + x
        v = v/norm(v)   
        W[k:, k] = v
        #Multiply the appropriate submatrix of A by the reflector
        R[k:,k:] -= 2*np.dot(v.reshape((dim, 1)), np.dot(v.reshape((1,dim)).conj(), R[k:,k:]))

    return W, R[:n, :n]

def formQ(W):
    """
    Make a reduced Q matrix from the v-s in W
    """
    m,n = W.shape
    Q = np.zeros((m,n), dtype = W.dtype)

    for k in range(n)[::-1]:
        Q[:,k] = ej(m, k)
        v = W[k:,k]
        dim = len(v)
        Q[k:,k:] -= 2*np.dot(v.reshape((dim,1)), np.dot(v.reshape((1,dim)).conj(), Q[k:,k:]))

    return Q

if __name__ == "__main__":
    m,n = 4, 3
    A = (1+1j)*np.arange(m*n)**2
    A = A.reshape((m,n))
    print "A:"
    print A
    W, R = house(A)
    print "W:"
    print W
    print "R:"
    print R
    Q = formQ(W)
    print "Q:"
    print Q
    print "QR=A? ", np.allclose(Q.dot(R), A)
