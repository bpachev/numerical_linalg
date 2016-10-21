from pivoted_QR import *
def test_algo():
    max_m = 10
    its = 10
    print "We test the pivoted QR algorithm by testing it for a variety of random matrices of all sizes."
    print "Maximum m={}, maximum tests per dimension {}".format(max_m, its)
    for m in xrange(3,max_m+1):
        for n in xrange(3, m+1):
            for i in xrange(its):
                A = np.random.random((m,n))
                p, Q, R = pivoted_QR(A)
                #assert that Q has orthonormal columns
                assert np.allclose(Q.T.dot(Q), np.eye(n))
                #assert that the diagonal entries of R decay in magnitude
                assert np.all(np.argsort(np.abs(np.diag(R))[::-1]) == np.arange(n))
                #assert R is upper triangular
                assert np.allclose(R, np.triu(R))
                #finally, assert that AP = QR, so the decomposition is valid
                if not np.allclose(A.T[p].T, Q.dot(R)):
                    print m,n
                    print Q.dot(R)
                    print A
                    raise ValueError("The algorithm FAILED")

if __name__ == "__main__":
    test_algo()
