import numpy as np
import sympy as sp

from .strain import StrainTensor
from .stress import StressTensor
from .material import ComplianceTensor, StiffnessTensor


class LinearElasticity:
    @staticmethod
    def hookes_law(
        compliance_tensor: ComplianceTensor, strain_tensor: StrainTensor
    ) -> StressTensor:
        # Voigt notation
        strain_tensor_data = strain_tensor.data * np.array([1, 1, 1, 2, 2, 2])
        return StressTensor(np.matmul(compliance_tensor.data, strain_tensor_data))

    @staticmethod
    def hookes_law_inverse(
        stiffness_tensor: StiffnessTensor, stress_tensor: StressTensor
    ) -> StrainTensor:
        strain_tensor_data = np.matmul(stiffness_tensor.data, stress_tensor.data)
        # Voigt notation
        strain_tensor_data = strain_tensor_data * np.array([1, 1, 1, 0.5, 0.5, 0.5])
        return StrainTensor(strain_tensor_data)
