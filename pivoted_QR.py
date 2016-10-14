import numpy as np
import numpy.linalg as la

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

def swap_rows(mat, i, j):
    tmp = mat[i].copy()
    mat[i] = mat[j]
    mat[j] = tmp

def formQ(W, swaps = None):
    """
    Make a reduced Q matrix from the v-s in W
    If swaps is not empty, it contains a list of integer indexes corresponding to permuatations.
    In particular, swaps[k] is the row number that was swapped with the k-th row at the k-th step.
    """
    m,n = W.shape
    Q = np.zeros((m,n), dtype = W.dtype)
    
    for k in range(n)[::-1]:
        Q[:,k] = ej(m, k)
        v = W[k:,k]
        dim = len(v)
        Q[k:,k:] -= 2*np.dot(v.reshape((dim,1)), np.dot(v.reshape((1,dim)).conj(), Q[k:,k:]))
        if swaps is not None:
            #We multiply on the right by the inverse of the swap matrix, which is just the same as the original swap because swapping twice is the indentity transformation
            swap_rows(Q, k, swaps[k])
    return Q

def formp(col_swaps):
    """
    Given the column swaps performed while pivoting, contruct the appropriate permutation of the columns of A 
    """
    n = len(col_swaps)
    p = np.arange(n)
    for k in reversed(range(n)):
        t = p[k]
        p[k] = col_swaps[k]
        p[col_swaps[k]] = t

    return p

def house(A):
    """
    Compute the implicit QR decomposition of a mxn complex numpy matrix A.
    Returns:
        W -- A lower-triangular complex mxn matrix, whose columns are the vs used for householder reflections.
        R -- The upper triangular nxn matrix in the QR decomposition.
    """
    m, n = A.shape
    if n > m: raise ValueError("The array must NOT have more columns than rows")

    R = A.copy()
    W = np.zeros(A.shape, dtype = R.dtype)

    for k in xrange(n):
        #select out the x-vector in the k-th column of R that needs to be reflected onto the span of e_1
        x = R[k:, k]
        dim = m-k
        #now pick an appropriate v for the Householder reflection
        v =  sign(x[0]) * la.norm(x) * ej(dim) + x
        v = v/norm(v)  
        W[k:, k] = v
        #Multiply the appropriate submatrix of A by the reflector
        R[k:,k:] -= 2*np.dot(v.reshape((dim, 1)), np.dot(v.reshape((1,dim)).conj(), R[k:,k:]))

    return W, R[:n, :n]

def pivoted_QR(A):
    """
    Description:
        A function to compute a pivoted QR factorization AP=QR, with the diagonal entries of R monotonically decreasing in magnitude.
        We compute this by a complete pivoting approach, where when operating on the submatrix  
    Arguments:
        A -- A complex valued mxn matrix, m >= n.
    Returns: (p,Q,R)
        p -- a vector containing the n permuted indices corresponding to the permutation matrix P
        Q -- an mxn matrix with orthonormal columns
        R -- an nxn upper triangular matrix with abs(R[0,0]) >= abs(R[1,1]) >= ... >= abs(R[-1, -1])        
    """

    m, n = A.shape
    dtype = A.dtype
    if not dtype == np.complex128: dtype = np.float64
    R = A[:n,:n].astype(dtype).copy()
    W = np.zeros(A.shape, dtype = R.dtype)
    row_swaps = np.zeros(n, dtype=int)
    col_swaps = np.zeros(n, dtype=int)
    for k in xrange(n):
        view = R[k:,k:]
        r, c = np.unravel_index(np.argmax(np.abs(R[k:,k:])), (m-k,n-k))
        row_swaps[k] = r+k
        col_swaps[k] = c+k
        tmp_row = view[r].copy()
        view[r] = view[0]
        view[0] = tmp_row
        tmp_col = view[:,c].copy()        
        view[:,c] = view[:,0]
        view[:,0] = tmp_col

        x = view[:,0]
#        print x
 #       print R
        dim = m-k
        #now pick an appropriate v for the Householder reflection
        v =  sign(x[0]) * la.norm(x) * ej(dim) + x
        v = v/la.norm(v)   
        W[k:, k] = v
        #Multiply the appropriate submatrix of A by the reflector
        R[k:,k:] -= 2*np.dot(v.reshape((dim, 1)), np.dot(v.reshape((1,dim)).conj(), R[k:,k:]))

    Q = formQ(W, swaps=row_swaps)
    p = formp(col_swaps)
    return p, Q, R
    
A = np.arange(9).reshape((3,3)) + np.eye(3)
print A           
p,q,r = pivoted_QR(A)        
print q.dot(r).T[p].T        
