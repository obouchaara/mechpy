import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor as SS3X3T


class SymbolicStrainTensor(SS3X3T):
    STRAIN_VOIGT_MAPPING = {
        0: 1,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
    }

    def __init__(self, data, name=None, notation=None):
        super().__init__(data, name=name, notation=notation)

    @classmethod
    def create(cls, name="\\epsilon", notation=None):
        strain_tensor = super().create(name, notation=notation)
        if strain_tensor.notation == "voigt":
            data = strain_tensor.data
            mapping = cls.STRAIN_VOIGT_MAPPING
            new_components = [data[key] * value for key, value in mapping.items()]
            strain_tensor = cls.from_list(new_components, notation=notation, name=name)
        return strain_tensor

    def to_general(self):
        if self.notation == "standard":
            if self.name:
                return SymbolicStrainTensor.create(
                    name=self.name,
                    notation="voigt",
                ).to_general()
            raise ValueError("Should have a name")
        elif self.notation == "voigt":
            mapping = self.STRAIN_VOIGT_MAPPING
            new_components = [self.data[key] / value for key, value in mapping.items()]
            return SS3X3T.from_list(new_components, self.notation).to_general()
        raise ValueError()

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def principal_components(self):
        return self.eigenvalues().keys()

    def volumetric_strain(self):
        return sum(self.normal_components())
