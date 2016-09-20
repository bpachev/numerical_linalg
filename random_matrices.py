import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def random_mat(m):
    """
    Return a random mxm matrix whose entries are drawn from the real normal distribution with mean 0, devation 1/sqrt(m)
    """
    return np.random.randn(m,m) / m**.5

def plot_svals(svals):
    svals = np.array(svals)
    m = -np.log(np.min(svals))/np.log(2)
    bins = [2**(i) for i in xrange(-int(m)-1, 1)]
#    print m, bins, svals
    plt.hist(svals, bins, normed = 1, cumulative=1)
    plt.xscale('log')
    plt.show()

#explore eigenvalues of random matrices
def square_mats():
    ms = []
    avg_rads = []
    avg_norms = []
    for exp in xrange(3, 7):
        m = 2**exp
        ms.append(m)
        rs = []
        two_norms = []
        smallest_singular = []
        for i in xrange(100):
            A = random_mat(m)
            two_norms.append(la.norm(A, ord=2))
            smallest_singular.append(la.norm(A, ord=-2))
            evals = la.eigvals(random_mat(m))
            rs.append(np.max(np.abs(evals)))
#            plt.scatter(np.real(evals), np.imag(evals))

        # plt.title("Eigenvalue distribution for random {}x{} matrices".format(m,m))
        # plt.axes().set_aspect('equal', 'datalim')
        # plt.show()
        plot_svals(smallest_singular)
        avg_rads.append(sum(rs)/len(rs))
        avg_norms.append(sum(two_norms)/len(rs))
    print "(a) It looks like the eigen-values fall in the complex unit ball! The spectral radius is around 1."
    plt.plot(ms, avg_rads, label="Average spectral radii")
    plt.plot(ms, avg_norms, label="Average 2-norms")
    plt.legend()
    plt.ylim(1,3)
    plt.xlabel("Dimension")
    plt.show()
    print "(b) It appears that the 2-norm is bounded in the limit, but larger than the spectral radius"


if __name__ == "__main__":
    square_mats()
