import numpy as np
import numpy.linalg as la

def svd_errors(fix_signs = False, e = 1, reps=5, n = 50):
    for i in xrange(reps):
        if i: print   
        print "Matrix {}".format(i+1)
        M = np.random.random((n, n))
        U1, S1, V1 = la.svd(M)
        S1 = np.sort(np.random.random((n)))[::-1]**e
        S1 = np.diag(S1)
        A = U1.dot(S1.dot(V1))
    
        U2, S2, V2 = la.svd(A)
        S2 = np.diag(S2)
        if fix_signs:
            signs =  np.diag(U1.T.dot(U2))
            U2 *= signs
            V2 = V2.T * signs
#        print np.diag(U2.T.dot(U1))
        
    #    print np.diag(V2.T.dot(V1))
        
        print "|A-U2*S2*V2| = ", la.norm(A-U2.dot(S2).dot(V2))
        print "|U-U2| = ",la.norm(U1-U2)," |U-U2| = ", la.norm(S1-S2), "|V-V2| = ",la.norm(V1.T-V2.T)
        print "cond(A) = ", la.cond(A)


if __name__ == "__main__":
      print "Part (a)"
      svd_errors()
      print "It appears that the svd is backwards stable but has high forward error in U and V."
      print 
      print "Part (b)"
      svd_errors(fix_signs = True)
      print "Changing the signs of U's columns fixes the error on U but not V."
      print
      print "Part (c)"
      svd_errors(fix_signs=True, e=6)
      print "High condition numbers on A worsen the forward error, unsurprisingly."
