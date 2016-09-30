import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def random_lower(n):
    """
    Make a random lower triangular matrix with diagonal entries equal to 1 and all other entries plus or minus 1
    """
    L = np.tril(np.random.random_integers(0,1,size=(n,n)) * 2 - 1)
    L[np.diag_indices(n)] = 1
    return L



if __name__ == "__main__":
    samples = 5
    m_list = [2**e for e in xrange(2,10)]
#    m_list = [2500]
    conds = []
    for m in m_list:
        conds.append(sum([la.cond(random_lower(m))**(1./m) for i in xrange(samples)])/ float(samples))
    print conds
#    plt.plot(m_list, conds)
 #   plt.xscale('log')
  #  plt.show()
 
