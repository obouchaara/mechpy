import sympy as sp

from .strain import SymbolicStrainTensor
from .stress import SymbolicStressTensor
from .material import SymbolicComplianceTensor, SymbolicStiffnessTensor


class SymbolicLinearElasticity:
    @staticmethod
    def hookes_law(
        compliance_tensor: SymbolicComplianceTensor, strain_tensor: SymbolicStrainTensor
    ) -> SymbolicStressTensor:
        pass

    @staticmethod
    def hookes_law_inverse(
        stiffness_tensor: SymbolicStiffnessTensor, stress_tensor: SymbolicStressTensor
    ) -> SymbolicStrainTensor:
        pass
