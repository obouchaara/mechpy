import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor


class SymbolicStrainTensor(SymbolicSymmetricThreeByThreeTensor):
    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def create(cls, name="\\epsilon", mode=0):
        return super().create(name, mode)

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def volumetric_strain(self):
        return sum(self.normal_components())
