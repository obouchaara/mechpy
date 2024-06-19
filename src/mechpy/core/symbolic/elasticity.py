import sympy as sp

from .strain import SymbolicStrainTensor
from .stress import SymbolicStressTensor
from .material import SymbolicComplianceTensor, SymbolicStiffnessTensor


class SymbolicLinearElasticity:
    """
    A class used to represent symbolic linear elasticity computations.

    Methods
    -------
    hookes_law(compliance_tensor, stress_tensor)
        Computes the strain tensor from the compliance tensor and the stress tensor.

    hookes_law_inverse(stiffness_tensor, strain_tensor)
        Computes the stress tensor from the stiffness tensor and the strain tensor.
    """

    @staticmethod
    def hookes_law(
        compliance_tensor: SymbolicComplianceTensor,
        stress_tensor: SymbolicStressTensor,
    ) -> SymbolicStrainTensor:
        """
        Computes the strain tensor from the compliance tensor and the stress tensor using Hooke's Law.

        :param compliance_tensor: SymbolicComplianceTensor
            The compliance tensor of the material.
        :param stress_tensor: SymbolicStressTensor
            The stress tensor.
        :return: SymbolicStrainTensor
            The computed strain tensor.
        """
        strain_tensor = compliance_tensor @ stress_tensor
        return strain_tensor

    @staticmethod
    def hookes_law_inverse(
        stiffness_tensor: SymbolicStiffnessTensor,
        strain_tensor: SymbolicStrainTensor,
    ) -> SymbolicStressTensor:
        """
        Computes the stress tensor from the stiffness tensor and the strain tensor using the inverse of Hooke's Law.

        :param stiffness_tensor: SymbolicStiffnessTensor
            The stiffness tensor of the material.
        :param strain_tensor: SymbolicStrainTensor
            The strain tensor.
        :return: SymbolicStressTensor
            The computed stress tensor.
        """
        stress_tensor = stiffness_tensor @ strain_tensor
        return stress_tensor
