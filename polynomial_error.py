import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb

if __name__ == "__main__":
    n = 9
    coeffs = [(-2)**i*comb(n, i, True) for i in xrange(n+1)]
    x = np.linspace(2-.08, 2+.08, 161)
    y1 = np.zeros(len(x))
    for i, c in enumerate(coeffs):
        y1 += c*x**i

    plt.subplot(211)
    plt.plot(x, y1)
    plt.title("(x-2)^9 evaluated from the coefficients")
    plt.subplot(212)
    plt.plot(x, (x-2)**n)
    plt.title("(x-2)^9 evaluated directly")
    plt.tight_layout()
    plt.show()
