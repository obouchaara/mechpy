import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor


class SymbolicStrainTensor(SymbolicSymmetricThreeByThreeTensor):
    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def create(cls, name="\\epsilon"):
        return super().create(name)

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def volumetric_strain(self):
        pass

    def shear_strain(self):
        pass