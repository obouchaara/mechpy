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
        compliance_matrix = sp.Matrix(compliance_tensor.data.tolist())
        strain_matrix = sp.Matrix(stress_tensor.data.tolist())
        strain_matrix = compliance_matrix @ strain_matrix
        strain_data = sp.ImmutableDenseNDimArray(strain_matrix, (6,))
        strain_tensor = SymbolicStressTensor(strain_data, notation=2)
        return strain_tensor

    @staticmethod
    def hookes_law_inverse(
        stiffness_tensor: SymbolicStiffnessTensor,
        strain_tensor: SymbolicStrainTensor,
    ) -> SymbolicStressTensor:
        stiffness_matrix = sp.Matrix(stiffness_tensor.data.tolist())
        strain_matrix = sp.Matrix(strain_tensor.data.tolist())
        stress_matrix = stiffness_matrix @ strain_matrix
        stress_data = sp.ImmutableDenseNDimArray(stress_matrix, (6,))
        stress_tensor = SymbolicStressTensor(stress_data, notation=2)
        return stress_tensor
