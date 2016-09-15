from mod_gram_schmidt import mgs
from householder import house, formQ
import numpy as np
from numpy.linalg import qr

Z = np.array(
[[1, 2, 3],
[4,5,6],[7,8,7],[4,3,2],[4,2,2]
])

if __name__ == "__main__":
    q1, r1 = mgs(Z)
    w, r2 = house(Z)
    q2 = formQ(w)

    q3, r3 = qr(Z)

    print "Is mgs close to householder for Q?", np.allclose(q1,q2)
    print "householder close to built-in for Q?", np.allclose(q3,q2)
    print "Is mgs close to householder for R?", np.allclose(r1,r2)
    print "householder close to built-in for R?", np.allclose(r3,r2)
