import copy
import unittest
import numpy as np
import sympy as sp

from mechpy.core.symbolic.coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)


class TestSymbolicCoordSystem(unittest.TestCase):
    def test_eq(self):
        origin_a = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis_a = sp.symbols("x1 x2 x3")
        origin_b = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis_b = sp.symbols("x1 x2 x3")
        coord_system_a = SymbolicCoordSystem(origin_a, basis_a)
        coord_system_b = SymbolicCoordSystem(origin_b, basis_b)
        self.assertEqual(coord_system_a, coord_system_b)
        
        origin = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis = sp.symbols("x1 x2 x3")
        coord_system_a = SymbolicCoordSystem(origin, basis)
        coord_system_b = copy.deepcopy(coord_system_a)
        self.assertEqual(coord_system_a, coord_system_b)
        
