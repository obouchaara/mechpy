import numpy as np
import sympy as sp

from .tensor import SymmetricThreeByThreeTensor


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
