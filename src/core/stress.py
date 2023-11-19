import numpy as np
import sympy as sp

from .tensor import SymmetricThreeByThreeTensor


class StressTensor(SymmetricThreeByThreeTensor):
    def __init__(self, data):
        # Call the constructor of the base class
        super().__init__(data)

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def pressure(self):
        return self.normal_components().sum() / 3.0

    def tresca(self):
        sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12 = self.data

        max_shear_stress = max(
            abs(sigma_11 - sigma_22),
            abs(sigma_22 - sigma_33),
            abs(sigma_33 - sigma_11),
            abs(sigma_23),
            abs(sigma_13),
            abs(sigma_12),
        )

        return max_shear_stress

    def von_mises(self):
        sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12 = self.data

        von_mises = np.sqrt(
            0.5
            * (
                (sigma_11 - sigma_22) ** 2
                + (sigma_22 - sigma_33) ** 2
                + (sigma_33 - sigma_11) ** 2
                + 6 * (sigma_23**2 + sigma_13**2 + sigma_12**2)
            )
        )

        return von_mises
