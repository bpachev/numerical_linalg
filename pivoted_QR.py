import numpy as np
import numpy.linalg as la

def pivoted_QR(A):
    """
    Description:
        A function to compute a pivoted QR factorization AP=QR, with the diagonal entries of R monotonically decreasing in magnitude.
    Arguments:
        A -- A complex valued mxn matrix, m >= n.
    Returns: (P,Q,R)
        P -- a containing the permuted indices corresponding to the permutation matrix P
        Q -- an mxn matrix with  
    """
