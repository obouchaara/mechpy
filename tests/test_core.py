import sys
import os
import unittest
import sympy as sp

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from core.elasticity.strain.symbolic import (
    to_strain_voigt_symbolic,
)


class TestCore(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
