import sympy as sp

from .material import (
    SymbolicComplianceTensor,
    SymbolicStiffnessTensor,
    SymbolicElasticMaterial,
)
from .displacement import SymbolicDisplacement
from .strain import SymbolicStrainTensor
from .stress import SymbolicStressTensor
from .navier import SymbolicNavier


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


class SymbolicElasticity:
    def __init__(
        self,
        material: SymbolicElasticMaterial = None,
        displacement: SymbolicDisplacement = None,
        strain_tensor: SymbolicStrainTensor = None,
        stress_tensor: SymbolicStressTensor = None,
        # solicitation: SymbolicSolicitation = None # to implement
    ):

        self.material = material
        self.displacement = displacement
        self.strain_tensor = strain_tensor
        self.stress_tensor = stress_tensor
        # self.solicitation = solicitation # to implement

    def get_strain_tensor(self):
        if self.material is None:
            raise ValueError(
                "Material is not set. Please set a valid SymbolicElasticMaterial."
            )
        if not isinstance(self.material, SymbolicElasticMaterial):
            raise TypeError(
                f"Expected SymbolicElasticMaterial, but got {type(self.material).__name__}"
            )
        if self.stress_tensor is None:
            raise ValueError(
                "Stress tensor is not set. Please set a valid SymbolicStressTensor."
            )
        if not isinstance(self.stress_tensor, SymbolicStressTensor):
            raise TypeError(
                f"Expected SymbolicStressTensor, but got {type(self.stress_tensor).__name__}"
            )
        return hookes_law(
            self.material.compliance_tensor(),
            self.stress_tensor,
        )

    def get_stress_tensor(self):
        if self.material is None:
            raise ValueError(
                "Material is not set. Please set a valid SymbolicElasticMaterial."
            )
        if not isinstance(self.material, SymbolicElasticMaterial):
            raise TypeError(
                f"Expected SymbolicElasticMaterial, but got {type(self.material).__name__}"
            )
        if self.strain_tensor is None:
            raise ValueError(
                "Strain tensor is not set. Please set a valid SymbolicStrainTensor."
            )
        if not isinstance(self.strain_tensor, SymbolicStrainTensor):
            raise TypeError(
                f"Expected SymbolicStrainTensor, but got {type(self.strain_tensor).__name__}"
            )
        return hookes_law_inverse(
            self.material.stiffness_tensor(),
            self.strain_tensor,
        )

    @property
    def navier(self):
        if self.material is None:
            raise ValueError(
                "Material is not set. Please set a valid SymbolicElasticMaterial."
            )
        if not isinstance(self.material, SymbolicElasticMaterial):
            raise TypeError(
                f"Expected SymbolicElasticMaterial, but got {type(self.material).__name__}"
            )

        if self.displacement is None:
            raise ValueError(
                "Displacement is not set. Please set a valid SymbolicDisplacement."
            )
        if not isinstance(self.displacement, SymbolicDisplacement):
            raise TypeError(
                f"Expected SymbolicDisplacement, but got {type(self.displacement).__name__}"
            )

        return SymbolicNavier(
            material=self.material,
            displacement=self.displacement,
        )

    def general_navier_equation(self):
        return self.navier.general_equation()

    def static_navier_equation(self):
        return self.navier.static_equation()
