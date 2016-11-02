import numpy as np
import numpy.linalg as la
from epsilon_pseudospectra import mk_A

def power(A, v, tol=1e-10, its=100):
    i = 0
    v = v/la.norm(v)
    while True:
        i += 1
        v = A.dot(v)
        v = v/la.norm(v)
        lam = np.dot(v, A.dot(v))
        if la.norm(A.dot(v) - lam*v) < tol: break 
        if i > its:
            break
    
    return lam, v

        
    
def inverse(A, v, mu=1, tol=1e-10, its=100):
    i = 0
    v = v/la.norm(v)
    I = np.eye(A.shape[0])
    while True:
        i += 1
        v = la.solve(A-mu*I, v)
        v = v/la.norm(v)
        lam = np.inner(v, A.dot(v))
        if la.norm(A.dot(v) - lam*v) < tol: break 
        if i > its:
            break
    
    return lam, v


def rayleigh(A, v, tol=1e-10, its=100):
    i = 0
    v = v/la.norm(v)
    I = np.eye(A.shape[0])
    lam = np.dot(v, A.dot(v))
    while True:
        i += 1
        v = la.solve(A-lam*I, v)
        v = v/la.norm(v)
        lam = np.dot(v, A.dot(v))
        error = la.norm(A.dot(v) - lam*v)
        if error < tol: break 
        if i > its:
            break
    
    return lam, v

if __name__ == "__main__":
    n = 32
    A = np.array(mk_A(n))
    evals, evecs = la.eigh(A)
    v0 = np.random.random(n)
#    i = np.argmax(np.abs(evals))
 #   val, vec = evals[i], evecs[:,i]
    print "We apply various algorithms to the matrix of 26.2"
    def print_error(lam, v):
        print "||Ax-lambda*x|| = {}".format(la.norm(A.dot(v) - lam*v))
    print "Power Method with a 1000 iterations"    
    print_error(*power(A, v0, its=1000))
    print "Rayleigh"
    print_error(*rayleigh(A, v0))
    print "Inverse"
    print_error(*inverse(A, v0))
    

    
