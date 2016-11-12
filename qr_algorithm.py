import matplotlib.pyplot as plt
import numpy as np
from householder import house, formQ, sign, ej
from scipy.linalg import norm, eig

def hilb(n):
    a = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            a[i,j] = 1./(i+j+1)
    return a

def tridiag(A):
    m = A.shape[0]
    R = A.copy()
    W = np.zeros(A.shape, dtype = R.dtype)
    for k in xrange(0,m-1):
        x = R[k+1:,k]
        dim = len(x)
        v = sign(x[0]) * norm(x) * ej(dim) + x
        v = v/norm(v)
        W[k+1:, k+1] = v
        #Multiply the appropriate submatrix of A by the reflector
        R[k+1:,k:] -= 2*np.dot(v.reshape((dim, 1)), np.dot(v.reshape((1,dim)).conj(), R[k+1:,k:]))
        R[:,k+1:] -= 2*np.outer(np.dot(R[:,k+1:], v), v.conj())

    return W, R

def wilkinson(B):
    """
    Compute the wilkinson shift given a 2x2 matrix B
    """
    am1, b, am = B[0,0], B[0,1], B[1,1]
    delt = (am1-am)/2.
    return am - sign(delt) * b**2/(abs(delt) + (delt**2+b**2)**.5)
    

def qralg(T, tol=1e-12, info=False, use_shift = False):
    Tnew = T
    its = 0
    I = np.eye(T.shape[0])
    ers = []
    while True:
        its += 1
        shift = wilkinson(T[-2:,-2:]) if use_shift else 0
        W, R = house(Tnew-I*shift)
        Q = formQ(W)
        Tnew = R.dot(Q).astype(np.float64) + I * shift
        err = abs(Tnew[-1, -2])
        ers.append(err)
        if err < tol:
            if not info: return Tnew
            else:   return Tnew, ers, its
        
def find_evals(A, plot_errs=True, title="", **kwargs):
    W, T = tridiag(A)
    m = T.shape[0]
    errs = []
    nits = 0
    evals = []
    for dim in xrange(m,1, -1):
        T = T[:dim, :dim]        
        T, ers, its = qralg(T,info=True, **kwargs)
        evals.append(T[-1,-1])
        errs.extend(ers)
        nits += its

    evals.append(T[0,0])
    plt.plot(range(1, nits+1), errs)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel("Number of iterations")
    plt.ylabel("Log error")
    plt.show()    
    return np.array(evals)
        

if __name__ == "__main__":
    A = hilb(4)
    find_evals(A, title="A = hilb(4), no shifts")
    evals = find_evals(A, title="A = hilb(4), shifts", use_shift = True)
    real = eig(A)[0]
    print "Are the computed eigenvalues close to the real ones? {}".format(np.allclose(np.sort(real), np.sort(evals)))

    A = np.diag(np.arange(15, 0, -1)) + np.ones((15,15))
    find_evals(A, title="part (e), no shifts")
    evals = find_evals(A, title="part (e), shifts", use_shift = True)
    real = eig(A)[0]
    print "Are the computed eigenvalues close to the real ones? {}".format(np.allclose(np.sort(real), np.sort(evals)))
    
    #print "

