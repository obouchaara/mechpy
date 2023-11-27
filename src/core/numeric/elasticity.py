import numpy as np

from .displacement import Displacement
from .strain import StrainTensor
from .stress import StressTensor
from .material import ComplianceTensor, StiffnessTensor


class LinearElasticity:
    @staticmethod
    def hookes_law(
        compliance_tensor: ComplianceTensor, strain_tensor: StrainTensor
    ) -> StressTensor:
        # Voigt notation
        correction_vector = np.array([1, 1, 1, 2, 2, 2]).reshape(6, 1)
        strain_tensor_data = strain_tensor.data * correction_vector
        return StressTensor(compliance_tensor.data @ strain_tensor_data)

    @staticmethod
    def hookes_law_inverse(
        stiffness_tensor: StiffnessTensor, stress_tensor: StressTensor
    ) -> StrainTensor:
        # Voigt notation correction
        correction_vector = np.array([1, 1, 1, 0.5, 0.5, 0.5]).reshape(6, 1)
        strain_tensor_data = stiffness_tensor.data @ stress_tensor.data
        corrected_strain_tensor_data = strain_tensor_data * correction_vector
        return StrainTensor(corrected_strain_tensor_data)

    @staticmethod
    def strain_from_displacement(displacement: Displacement) -> StrainTensor:
        displacement_gradient = displacement.gradient()
        strain = 0.5 * (displacement_gradient + np.transpose(displacement_gradient, axes=(0, 2, 1)))
        return StrainTensor(strain)