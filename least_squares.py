import numpy as np
import numpy.linalg as la
from scipy.special import gamma
import matplotlib.pyplot as plt

def approx_and_plot(a=1,b=2, npoints=200):
    if a == 0: a += 1e-10
    x = np.linspace(a,b, npoints)
    b = 1./x

    A = np.zeros((npoints, 3))
    A[:,0] = np.exp(x)
    A[:,1] = np.sin(x)
    A[:,2] = gamma(x)

    c, residuals, rank, svals = la.lstsq(A, b) #FYI the other return values are residuals, rank, and singular values
    error = residuals[0]
    print "L-2 norm error: ", error
    #the * is a sneaky trick equivalent to passing c[0], c[1], and c[2] as arguments
    print "Approximately, 1/x = {:.2f}*e^x + {:.2f}*sin(x) + {:.2f}*gamma(x)".format(*c)

    plt.plot(x, A.dot(c), label="Approximation")
    plt.plot(x, b, label="1/x")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    approx_and_plot(1,2)
    approx_and_plot(0,1)
