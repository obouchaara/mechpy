import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor as SS3X3T


class SymbolicStrainTensor(SS3X3T):
    STRAIN_VOIGT_MAPPING = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}

    def __init__(self, data, mode=0):
        if mode in [0, 1]:
            super().__init__(data)
            self.mode = mode
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

    @classmethod
    def create(cls, name="\epsilon", mode=0):
        if mode == 0:
            strain_tensor = super().create(name, mode=0)
        elif mode == 1:
            strain_tensor = super().create(name, mode=1)
            data = strain_tensor.data
            mapping = cls.STRAIN_VOIGT_MAPPING
            components = [data[key] * value for key, value in mapping.items()]
            new_data = sp.ImmutableDenseNDimArray(components)
            strain_tensor = cls(new_data, 1)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        return strain_tensor

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def volumetric_strain(self):
        return sum(self.normal_components())

    def to_general(self):
        if self.mode == 1:
            data = self.data
            mapping = self.STRAIN_VOIGT_MAPPING
            components = [data[k] / v for k, v in mapping.items()]
            return SS3X3T.from_list(components).to_general()
        else:
            raise NotImplementedError(f"Only implemented for mode 1")
