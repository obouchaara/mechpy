import sympy as sp

from .strain import SymbolicStrainTensor
from .stress import SymbolicStressTensor
from .material import SymbolicComplianceTensor, SymbolicStiffnessTensor


class SymbolicLinearElasticity:
    @staticmethod
    def hookes_law(
        compliance_tensor: SymbolicComplianceTensor,
        stress_tensor: SymbolicStressTensor,
    ) -> SymbolicStrainTensor:
        strain_tensor = compliance_tensor @ stress_tensor
        return strain_tensor

    @staticmethod
    def hookes_law_inverse(
        stiffness_tensor: SymbolicStiffnessTensor,
        strain_tensor: SymbolicStrainTensor,
    ) -> SymbolicStressTensor:
        stress_tensor = stiffness_tensor @ strain_tensor
        return stress_tensor
