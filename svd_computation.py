import matplotlib.pyplot as plt
from numpy.linalg import eigh, svd, norm
import numpy as np

#For problem 31.4 in Trefethen
if __name__ == "__main__":
    n = 30
    ns = range(1,n+1)
    #Computed using the stable magic algorithm from nUmpy
    real_sigs = []
    #Computed directly from eigenvalues of A.T*A
    bad_sigs = []
    for n in ns:
        A = np.triu(np.ones((n,n)))
        A -= .9 * np.eye(n)
        sigs = svd(A, compute_uv=False)
        real_sigs.append(np.min(sigs))
        squared = eigh(A.T.dot(A))[0]
        bad_sigs.append(np.sqrt(np.min(squared)))

    plt.plot(ns, real_sigs, label="True singular values")
    plt.plot(ns, bad_sigs, label="Unstably computed")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Dimension")
    plt.show()
