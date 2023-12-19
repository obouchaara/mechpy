import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor


class SymbolicStrainTensor(SymbolicSymmetricThreeByThreeTensor):
    STRAIN_VOIGT_MAPPING = {
        "\\epsilon_1": sp.symbols("\\epsilon_11"),
        "\\epsilon_2": sp.symbols("\\epsilon_22"),
        "\\epsilon_3": sp.symbols("\\epsilon_33"),
        "\\epsilon_4": sp.symbols("\\epsilon_23") * 2,
        "\\epsilon_5": sp.symbols("\\epsilon_13") * 2,
        "\\epsilon_6": sp.symbols("\\epsilon_12") * 2,
    }

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

    def to_general(self):
        general_tensor = super().to_general()
        general_tensor.data = general_tensor.data.subs(self.STRAIN_VOIGT_MAPPING)
        return general_tensor
