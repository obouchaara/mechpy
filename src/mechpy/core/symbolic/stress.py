import sympy as sp

from .tensor import SymbolicSymmetricThreeByThreeTensor as SS3X3T


class SymbolicStressTensor(SS3X3T):
    STRESS_VOIGT_MAPPING = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    def __init__(self, data, name=None, notation=1):
        super().__init__(data, name=name, notation=notation)

    @classmethod
    def create(cls, name="\sigma", notation=1):
        stress_tensor = super().create(name, notation)
        if notation == 2:
            data = stress_tensor.data
            mapping = cls.STRESS_VOIGT_MAPPING
            new_components = [data[key] * value for key, value in mapping.items()]
            stress_tensor = cls.from_list(new_components, notation=notation, name=name)
        return stress_tensor

    def to_general(self):
        if self.notation == 1:
            if self.name:
                return SymbolicStressTensor.create(
                    name=self.name, notation=2
                ).to_general()
            raise ValueError("Should have a name")
        elif self.notation == 2:
            data = self.data
            mapping = self.STRESS_VOIGT_MAPPING
            new_components = [data[key] / value for key, value in mapping.items()]
            return SS3X3T.from_list(new_components, self.notation).to_general()
        raise ValueError()

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]
    
    def principal_components(self):
        return self.eigenvalues().keys()

    def pressure(self):
        return sum(self.normal_components()) / sp.symbols("3")

    def tresca(self):
        principal_stresses = self.principal_components()
        components = [
            sp.Abs(stress1 - stress2)
            for stress1 in principal_stresses
            for stress2 in principal_stresses
        ]
        max_diff = sp.Max(*components)
        return max_diff

    def von_mises(self):
        s_11, s_22, s_33, s_23, s_13, s_12 = self.data

        first_term = ((s_11 - s_22) ** 2 + (s_22 - s_33) ** 2 + (s_33 - s_11) ** 2) / 2
        second_term = 3 * (s_23**2 + s_13**2 + s_12**2)
        von_mises_stress = sp.sqrt(first_term + second_term)

        return von_mises_stress
