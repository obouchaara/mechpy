import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor as SS3X3T


class SymbolicStrainTensor(SS3X3T):
    STRAIN_VOIGT_MAPPING = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}

    def __init__(self, data, notation=1):
        super().__init__(data, notation)

    @classmethod
    def create(cls, name="\epsilon", notation=1):
        strain_tensor = super().create(name, notation)
        if notation == 2:
            data = strain_tensor.data
            mapping = cls.STRAIN_VOIGT_MAPPING
            new_components = [data[key] * value for key, value in mapping.items()]
            strain_tensor = cls.from_list(new_components, 2)
        return strain_tensor

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def volumetric_strain(self):
        return sum(self.normal_components())

    def to_general(self):
        if self.notation != 2:
            name = self.NOTATIONS[2]["name"]
            raise NotImplementedError(f"Only implemented for {name}")
        data = self.data
        mapping = self.STRAIN_VOIGT_MAPPING
        new_components = [data[key] / value for key, value in mapping.items()]
        return SS3X3T.from_list(new_components, self.notation).to_general()
