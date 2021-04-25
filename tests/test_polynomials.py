import unittest
import numpy as np
import freenn.utils.polynomials as polynomials

class TestPolynomialsMethods(unittest.TestCase):

    def test_pascal_triangle(self):
        pascal_triangle = polynomials.pascal(4)
        coeffs = pascal_triangle[:,0]
        self.assertTrue( (coeffs == [1, 3, 3, 1]).all()  )

    def test_taylor_expand(self):
        # P(X) = (X-1)^3
        p = np.array([1,-3, 3, -1]).transpose()
        # Test P(1) = 0
        self.assertEqual( np.polyval(p, 1), 0 )
        # Test coeffs of P(X+2)
        coeffs = polynomials.taylor_expand(p, 2)
        self.assertTrue( (coeffs == [1, 3, 3, 1]).all()  )

if __name__ == '__main__':
    unittest.main()