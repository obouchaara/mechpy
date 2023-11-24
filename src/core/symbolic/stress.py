import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor


class SymbolicStressTensor(SymbolicSymmetricThreeByThreeTensor):
    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def create(cls, name="\\sigma"):
        return super().create(name)

    def normal_components(self):
        return sp.Matrix(self.data[:3])

    def shear_components(self):
        return sp.Matrix(self.data[3:])

    def pressure(self):
        return sum(self.normal_components()) / 3.0
