import numpy as np
import sympy as sp
import unittest

from mechpy.core import (
    SymbolicThreeByThreeTensor,
    SymbolicSymmetricThreeByThreeTensor,
)
from mechpy.core import (
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)


class TestCore(unittest.TestCase):
    pass


class TestSymbolicThreeByThreeTensor(unittest.TestCase):
    def test_initialization_valid_data(self):
        valid_array = sp.ImmutableDenseNDimArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = SymbolicThreeByThreeTensor(valid_array)
        self.assertEqual(tensor.data, valid_array)

    def test_initialization_invalid_data(self):
        invalid_matrix = sp.ImmutableDenseNDimArray([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            SymbolicThreeByThreeTensor(invalid_matrix)

    def test_to_symmetric_valid(self):
        symmetric_array = sp.ImmutableDenseNDimArray([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        tensor = SymbolicThreeByThreeTensor(symmetric_array)
        symmetric_tensor = tensor.to_symmetric()
        self.assertIsInstance(symmetric_tensor, SymbolicSymmetricThreeByThreeTensor)
        self.assertTrue(symmetric_tensor.is_symmetric())

    def test_to_symmetric_invalid(self):
        non_symmetric_array = sp.ImmutableDenseNDimArray(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        )
        tensor = SymbolicThreeByThreeTensor(non_symmetric_array)
        with self.assertRaises(ValueError):
            tensor.to_symmetric()


class TestSymbolicSymmetricThreeByThreeTensor(unittest.TestCase):
    def test_notation_standard_map(self):
        expected_maps = {
            2: {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2},
            3: {
                (0, 0): 0,
                (0, 1): 1,
                (0, 2): 2,
                (1, 0): 1,
                (1, 1): 3,
                (1, 2): 4,
                (2, 0): 2,
                (2, 1): 4,
                (2, 2): 5,
            },
            4: {
                (0, 0): 0,
                (0, 1): 1,
                (0, 2): 2,
                (0, 3): 3,
                (1, 0): 1,
                (1, 1): 4,
                (1, 2): 5,
                (1, 3): 6,
                (2, 0): 2,
                (2, 1): 5,
                (2, 2): 7,
                (2, 3): 8,
                (3, 0): 3,
                (3, 1): 6,
                (3, 2): 8,
                (3, 3): 9,
            },
        }

        for n, expected_map in expected_maps.items():
            assert (
                SymbolicSymmetricThreeByThreeTensor.notation_standard_map(n)
                == expected_map
            )

    def test_notation_standard_inverse_map(self):
        expected_inverse_maps = {
            2: {0: (0, 0), 1: (0, 1), 2: (1, 1)},
            3: {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 1), 4: (1, 2), 5: (2, 2)},
            4: {
                0: (0, 0),
                1: (0, 1),
                2: (0, 2),
                3: (0, 3),
                4: (1, 1),
                5: (1, 2),
                6: (1, 3),
                7: (2, 2),
                8: (2, 3),
                9: (3, 3),
            },
        }

        for n, expected_inverse_map in expected_inverse_maps.items():
            assert (
                SymbolicSymmetricThreeByThreeTensor.notation_standard_inverse_map(n)
                == expected_inverse_map
            )


class TestSymbolicCartesianCoordSystem(unittest.TestCase):
    def test_get_basis_cylindrical_exprs(self):
        # Create an instance of SymbolicCartesianCoordSystem
        cartesian_system = SymbolicCartesianCoordSystem()

        # Call get_basis_cylindrical_exprs method
        expr_dict = cartesian_system.get_basis_cylindrical_exprs()

        # Check if the dictionary has the correct keys
        self.assertIn("r", expr_dict)
        self.assertIn("theta", expr_dict)
        self.assertIn("z", expr_dict)

        # Check if the expressions are correct
        x, y, z = sp.symbols("x y z")
        expected_r_expr = sp.sqrt(x**2 + y**2)
        expected_theta_expr = sp.atan2(y, x)
        expected_z_expr = z

        self.assertEqual(expr_dict["r"], expected_r_expr)
        self.assertEqual(expr_dict["theta"], expected_theta_expr)
        self.assertEqual(expr_dict["z"], expected_z_expr)

    def test_get_basis_spherical_exprs(self):
        # Create an instance of SymbolicCartesianCoordSystem
        cartesian_system = SymbolicCartesianCoordSystem()

        # Call get_basis_spherical_exprs method
        expr_dict = cartesian_system.get_basis_spherical_exprs()

        # Check if the dictionary has the correct keys
        self.assertIn("r", expr_dict)
        self.assertIn("theta", expr_dict)
        self.assertIn("phi", expr_dict)

        # Check if the expressions are correct
        x, y, z = sp.symbols("x y z")
        expected_r_expr = sp.sqrt(x**2 + y**2 + z**2)
        expected_theta_expr = sp.atan2(y, x)
        expected_phi_expr = sp.acos(z / expected_r_expr)

        self.assertEqual(expr_dict["r"], expected_r_expr)
        self.assertEqual(expr_dict["theta"], expected_theta_expr)
        self.assertEqual(expr_dict["phi"], expected_phi_expr)

    def test_to_cylindrical(self):
        # Create an instance of SymbolicCartesianCoordSystem
        cartesian_system = SymbolicCartesianCoordSystem()

        # Convert to cylindrical coordinate system
        cylindrical_system = cartesian_system.to_cylindrical()

        # Check if the returned object is an instance of SymbolicCylindricalCoordSystem
        self.assertIsInstance(cylindrical_system, SymbolicCylindricalCoordSystem)

        # Check if the basis symbols are correct
        x, y, z = sp.symbols("x y z")
        expected_r_expr = sp.sqrt(x**2 + y**2)
        expected_theta_expr = sp.atan2(y, x)
        expected_z_expr = z

        self.assertEqual(cylindrical_system.basis_symbols[0], expected_r_expr)
        self.assertEqual(cylindrical_system.basis_symbols[1], expected_theta_expr)
        self.assertEqual(cylindrical_system.basis_symbols[2], expected_z_expr)

    def test_to_spherical(self):
        # Create an instance of SymbolicCartesianCoordSystem
        cartesian_system = SymbolicCartesianCoordSystem()

        # Convert to spherical coordinate system
        spherical_system = cartesian_system.to_spherical()

        # Check if the returned object is an instance of SymbolicSphericalCoordSystem
        self.assertIsInstance(spherical_system, SymbolicSphericalCoordSystem)

        # Check if the basis symbols are correct
        x, y, z = sp.symbols("x y z")
        expected_r_expr = sp.sqrt(x**2 + y**2 + z**2)
        expected_theta_expr = sp.atan2(y, x)
        expected_phi_expr = sp.acos(z / expected_r_expr)

        self.assertEqual(spherical_system.basis_symbols[0], expected_r_expr)
        self.assertEqual(spherical_system.basis_symbols[1], expected_theta_expr)
        self.assertEqual(spherical_system.basis_symbols[2], expected_phi_expr)

    def test_get_cylindrical_coord(self):
        # Create an instance of SymbolicCartesianCoordSystem
        cartesian_system = SymbolicCartesianCoordSystem()

        # Define a set of Cartesian coordinates (x, y, z)
        x_val, y_val, z_val = 3, 4, 5  # Example values

        # Call get_cylindrical_coord method with these values
        cylindrical_coords = cartesian_system.get_cylindrical_coord(
            (x_val, y_val, z_val)
        )

        # Calculate expected cylindrical coordinates
        expected_r = sp.sqrt(x_val**2 + y_val**2)
        expected_theta = sp.atan2(y_val, x_val)
        expected_z = z_val

        # Check if the returned cylindrical coordinates are as expected
        self.assertEqual(cylindrical_coords[0], expected_r)
        self.assertEqual(cylindrical_coords[1], expected_theta)
        self.assertEqual(cylindrical_coords[2], expected_z)

    def test_get_spherical_coord(self):
        # Create an instance of SymbolicCartesianCoordSystem
        cartesian_system = SymbolicCartesianCoordSystem()

        # Define a set of Cartesian coordinates (x, y, z)
        x_val, y_val, z_val = 3, 4, 5  # Example values

        # Call get_spherical_coord method with these values
        spherical_coords = cartesian_system.get_spherical_coord((x_val, y_val, z_val))

        # Calculate expected spherical coordinates
        expected_r = sp.sqrt(x_val**2 + y_val**2 + z_val**2)
        expected_theta = sp.atan2(y_val, x_val)
        expected_phi = sp.acos(z_val / expected_r)

        # Check if the returned spherical coordinates are as expected
        self.assertEqual(spherical_coords[0], expected_r)
        self.assertEqual(spherical_coords[1], expected_theta)
        self.assertEqual(spherical_coords[2], expected_phi)


if __name__ == "__main__":
    unittest.main()
