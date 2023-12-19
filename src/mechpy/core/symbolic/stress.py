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
    
    def tresca(self):
        principal_stresses = self.to_general().data.eigenvals()

        principal_stresses = sp.Matrix(list(principal_stresses.keys()))

        max_diff = sp.Max(*[sp.Abs(stress1 - stress2) for stress1 in principal_stresses for stress2 in principal_stresses])

        return max_diff

    def von_mises(self):
        sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12 = self.data

        # Von Mises Stress calculation for a symmetric stress tensor
        von_mises_stress = sp.sqrt(
            ((sigma_11 - sigma_22)**2 + (sigma_22 - sigma_33)**2 + (sigma_33 - sigma_11)**2) / 2
            + 3 * (sigma_23**2 + sigma_13**2 + sigma_12**2)
        )

        return von_mises_stress

