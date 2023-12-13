import sys
import numpy as np
import sympy as sp
import unittest

from mechpy.core import SymbolicThreeByThreeTensor, SymbolicSymmetricThreeByThreeTensor


class TestCore(unittest.TestCase):
    def test_initialization_valid_data(self):
        # Test initializing with valid data
        valid_matrix = sp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = SymbolicThreeByThreeTensor(valid_matrix)
        self.assertEqual(tensor.data, valid_matrix)

    def test_initialization_invalid_data(self):
        # Test initializing with invalid data (should raise ValueError)
        invalid_matrix = sp.Matrix([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            SymbolicThreeByThreeTensor(invalid_matrix)

    def test_to_symmetric_valid(self):
        # Test converting to symmetric tensor when valid
        # Assuming `SymbolicSymmetricThreeByThreeTensor` works correctly
        symmetric_matrix = sp.Matrix([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        tensor = SymbolicThreeByThreeTensor(symmetric_matrix)
        symmetric_tensor = tensor.to_symmetric()
        self.assertIsInstance(symmetric_tensor, SymbolicSymmetricThreeByThreeTensor)
        self.assertTrue(symmetric_tensor.is_symmetric())

    def test_to_symmetric_invalid(self):
        # Test converting to symmetric tensor when invalid (should raise ValueError)
        non_symmetric_matrix = sp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = SymbolicThreeByThreeTensor(non_symmetric_matrix)
        with self.assertRaises(ValueError):
            tensor.to_symmetric()


if __name__ == "__main__":
    unittest.main()
