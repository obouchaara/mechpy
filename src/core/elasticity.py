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
        return StressTensor(np.matmul(compliance_tensor.data, strain_tensor.data))

    @staticmethod
    def hookes_law_inverse(
        stiffness_tensor: StiffnessTensor, stress_tensor: StressTensor
    ) -> StrainTensor:
        return StrainTensor(np.matmul(stiffness_tensor.data, stress_tensor.data))
