import numpy as np

from .tensor import SymmetricThreeByThreeTensor
from .field import Field


class StressTensor(SymmetricThreeByThreeTensor):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"StressTensor(\n{self.data}\n)"

    def normal_components(self):
        return self.data[:3]

    def shear_components(self):
        return self.data[3:]

    def pressure(self):
        return self.normal_components().sum() / 3.0

    def tresca(self):
        stress_tensor = self.to_general().data
        principal_stresses = np.linalg.eigvals(stress_tensor)
        tresca_stress = np.max(principal_stresses) - np.min(principal_stresses)
        return tresca_stress

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
    

class StressField(Field):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"StressField(\n{self.data}\n)"
    
    def get(self, position) -> StressTensor:
        pass
