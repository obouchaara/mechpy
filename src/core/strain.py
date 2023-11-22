import numpy as np
import sympy as sp

from .tensor import SymmetricThreeByThreeTensor, SymbolicSymmetricThreeByThreeTensor


class StrainTensor(SymmetricThreeByThreeTensor):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"StrainTensor(\n{self.data}\n)"

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def volumetric_strain(self):
        pass

    def shear_strain(self):
        pass


class SymbolicStrainTensor(SymbolicSymmetricThreeByThreeTensor):
    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def create(cls):
        epsilon_11, epsilon_22, epsilon_33, epsilon_23, epsilon_13, epsilon_12 = sp.symbols(
            "epsilon_11 epsilon_22 epsilon_33 epsilon_23 epsilon_13 epsilon_12"
        )
        epsilon = sp.Matrix([epsilon_11, epsilon_22, epsilon_33, epsilon_23, epsilon_13, epsilon_12])
        return cls(epsilon)

    def normal_components(self):
        pass

    def shear_components(self):
        pass

    def volumetric_strain(self):
        pass

    def shear_strain(self):
        pass
