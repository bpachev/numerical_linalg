import numpy as np
import numpy.linalg as la
from scipy.special import gamma
import matplotlib.pyplot as plt

def approx_and_plot(a=1,b=2, npoints=100):
    x = np.linspace(a,b, npoints)
    target = 1./x

    A = np.zeros((npoints, 3))
    A[:,0] = np.exp(x)
    A[:,1] = np.sin(x)
    A[:,2] = gamma(x)

    c, residuals, rank, svals = la.lstsq(A, target) #FYI the other return values are residuals, rank, and singular values
    error = residuals[0]
    print "Approximating on the interval [{:.1f},{:.1f}] with {} points".format(a,b, npoints)
    print "Residual error: ", error
    #the * is a sneaky trick equivalent to passing c[0], c[1], and c[2] as arguments
    print "Approximately, 1/x = {:.2f}*e^x + {:.2f}*sin(x) + {:.2f}*gamma(x)".format(*c)

    plt.plot(x, A.dot(c), label="Approximation")
    plt.plot(x, target, label="1/x")
    plt.legend()
    plt.title("Approximation of 1/x on [{:.1f},{:.1f}] with {} points".format(a,b, npoints))
    plt.show()

if __name__ == "__main__":
    approx_and_plot(1,2)
    approx_and_plot(0.1,1, npoints=50)
