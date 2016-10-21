from pivoted_QR import *
from weighted_least_squares import do_plot, read_google_spreadsheet
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ra, nu = read_google_spreadsheet()
    dom = np.log(ra)
    n = len(dom)
    A = np.empty((n,2))
    A[:,0] = dom
    A[:,1] = 1
    b = np.log(nu)
    x = pivoted_least_squares(A,b)

    do_plot(ra,np.exp(A.dot(x)), nu, "Pivoted QR fit for unweighted problem")
    plt.show()
