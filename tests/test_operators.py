import sys
import os
import unittest
import sympy as sp

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

class TestTest(unittest.TestCase):
    def test_test(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
