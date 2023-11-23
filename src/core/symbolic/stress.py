import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor


class SymbolicStressTensor(SymbolicSymmetricThreeByThreeTensor):
    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def create(cls):
        sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12 = sp.symbols(
            "sigma_11 sigma_22 sigma_33 sigma_23 sigma_13 sigma_12"
        )
        sigma = sp.Matrix([sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12])
        return cls(sigma)

    def normal_components(self):
        return sp.Matrix(self.data[:3])

    def shear_components(self):
        return self.data[3:]

    def pressure(self):
        return sum(self.normal_components()) / 3.0
