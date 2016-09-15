import numpy as np

def mgs(A):
    """
    Compute a QR factorization of the mxn complex-valued matrix A, with m>=n.
    Assumes A is full-rank.
    Returns:
      Q -- an mxn complex valued matrix with orthonormal columns.
      R -- an nxn complex upper triangular matrix.
    """

    m,n = A.shape
    if n > m:
        raise ValueError("The matrix A may not have more columns than rows.")

    Q = A.astype(np.complex128)
    R = np.zeros((n,n), dtype=np.complex128)

    for i in xrange(n):
        R[i,i] = np.sqrt(np.vdot(Q[:,i], Q[:,i]))
        Q[:,i] /= R[i,i]

        for j in xrange(i+1,n):
            R[i,j] = np.vdot(Q[:,i], Q[:,j])
            Q[:,j] -= R[i,j] * Q[:,i]

    return Q, R

if __name__=="__main__":
    m,n = 8, 4
    A = np.random.rand(m,n)
    print "A, a random {}x{} matrix:".format(m,n)
    print A
    Q,R = mgs(A)
    print "Q:\n", Q
    print "R:\n", R
    print "Does QR=A within machine error?",np.allclose(A,Q.dot(R))
