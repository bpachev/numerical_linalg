import numpy as np
import numpy.linalg as la
from csv import reader
from sys import exit
import matplotlib.pyplot as plt

def weighted_diag_lstsq(A,b, C):
    """
    Solve the weighted least squares problem.
    A -- a full column rank mxn matrix
    b -- an m-vector
    C -- an m-vector consisting of the diagonal of the diagonal weight matrix
        C must contain only positive entries

    Returns x, the least squares solution
    """
    m = len(b)
    d = np.sqrt(C)
    #Multiply the rows of A by the entries of d
    B = A * d.reshape((m,1))
    b_hat = b*d
    return la.lstsq(B, b_hat)[0]

def do_plot(dom, fit, nu, title):
    plt.plot(dom, fit, label="Fit")
    plt.plot(dom, nu, label="Nu")
    plt.xlabel("Ra")
    plt.legend(loc="upper left")
    plt.title(title)

if __name__ == "__main__":
    filename = "prob2.csv"
    try:
        f = open(filename, "r")
    except IOError:
        print "Error opening file "+filename+" Check that the file exists and that you have read access."
        exit()

    csv_reader = reader(f)
    i = 0
    ra, nu = [], []
    for row in csv_reader:
        if i:
            ra.append(float(row[0]))
            nu.append(float(row[3]))
        i += 1

    ra = np.array(ra)
    nu = np.array(nu)

    #we compute a fit nu = a*ra^b
    #Taking logs naturally gives log(nu) = log(a) + log(ra)*log(b)
    n = len(ra)
    dom = np.log(ra)
    A = np.empty((n,2))
    A[:,0] = dom
    A[:,1] = 1
    b = np.log(nu)

    x = la.lstsq(A,b)[0]
    w1 = weighted_diag_lstsq(A,b,ra)
    w2 = weighted_diag_lstsq(A,b,ra**.1)
    w3 = weighted_diag_lstsq(A,b,np.log(ra))

    dom = ra
    plt.subplot(221)
    do_plot(dom, np.exp(A.dot(x)), nu, "Unweighted least squares fit")

    plt.subplot(222)
    do_plot(dom, np.exp(A.dot(w1)), nu, "Weighted, C=ra")

    plt.subplot(223)
    do_plot(dom, np.exp(A.dot(w2)), nu, "Weighted, C=ra^.1")

    plt.subplot(224)
    do_plot(dom, np.exp(A.dot(w3)), nu, "Weighted, C=log(ra)")
    plt.tight_layout()
    plt.show()

    print "Observe that greater weights on higher values of ra lead to better fits."

#    a,b = np.exp(x)
    #print ra, nu
