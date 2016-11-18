import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import diags
from arnoldi import arnoldi

def gmres(Amul, b, nits=1):
    Q, H = arnoldi(Amul, b, nits)
    residuals = []
    normb = la.norm(b)
    for n in xrange(1, nits+1):
        e1 = np.zeros(n+1)
        e1[0] = normb
        y, res = la.lstsq(H[:n+1, :n], e1)[:2]
        residuals.append(res[0])
        
    return Q[:,:nits].dot(y), residuals

if __name__ == "__main__":
    n = 50
    A = np.random.random((n,n))
    b = np.random.random(n)
    amul = lambda x: A.dot(x)
    sol, res = gmres(amul, b, n)
    plt.plot(res)
    plt.title("GMRES convergence on random {}x{} system".format(n,n))
    plt.ylabel("Residual")
    plt.xlabel("Iterations")
    plt.show()
