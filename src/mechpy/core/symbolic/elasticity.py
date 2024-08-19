import sympy as sp

from .coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)
from .material import (
    SymbolicComplianceTensor,
    SymbolicStiffnessTensor,
    SymbolicElasticMaterial,
)
from .displacement import SymbolicDisplacement
from .strain import SymbolicStrainTensor
from .stress import SymbolicStressTensor
from .solicitation import SymbolicVolumeForce
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
        coord_system: SymbolicCoordSystem = None,
        material: SymbolicElasticMaterial = None,
        displacement: SymbolicDisplacement = None,
        strain_tensor: SymbolicStrainTensor = None,
        stress_tensor: SymbolicStressTensor = None,
        volume_force: SymbolicVolumeForce = None,
        # boundary_conditions: list[SymbolicBoundaryCondition] = None
    ):
        if coord_system is None:
            coord_system = SymbolicCartesianCoordSystem()
        self.coord_system = coord_system
        
        if material is not None:
            self.set_material(material)
        else:
            self.material = None
            
        if displacement is not None:
            self.set_displacement(displacement)
        else:
            self.displacement = None
            
        if strain_tensor is not None:
            self.set_strain_tensor(strain_tensor)
        else:
            self.strain_tensor = None
            
        if stress_tensor is not None:
            self.set_stress_tensor(stress_tensor)
        else:
            self.stress_tensor = None
            
        if volume_force is not None:
            self.set_volume_force(volume_force)
        else:
            self.volume_force = None
        
        # if boundary_condition is not None:
        #     self.set_boundary_condition(boundary_condition)

    def set_material(self, material: SymbolicElasticMaterial):
        self.material = material

    def set_displacement(self, displacement: SymbolicDisplacement):
        if self.coord_system != displacement.coord_system:
            raise ValueError("Displacement's coordinate system does not match the elasticity problem's coordinate system.")
        self.displacement = displacement

    def set_strain_tensor(self, strain_tensor: SymbolicStrainTensor):
        self.strain_tensor = strain_tensor

    def set_stress_tensor(self, stress_tensor: SymbolicStressTensor):
        self.stress_tensor = stress_tensor

    def set_volume_force(self, volume_force: SymbolicVolumeForce):
        self.volume_force = volume_force

    # def set_boundary_condition(self, boundary_condition: SymbolicBoundaryCondition):
    #     self.boundary_conditions.append(boundary_condition)

    def compute_strain(self) -> None:
        if self.strain_tensor is not None:
            raise ValueError("Strain tensor is already set/computed.")
        
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
                "Stress tensor is not set/computed."
            )
        if not isinstance(self.stress_tensor, SymbolicStressTensor):
            raise TypeError(
                f"Expected SymbolicStressTensor, but got {type(self.stress_tensor).__name__}"
            )
            
        strain_tensor = hookes_law(
            self.material.compliance_tensor(),
            self.stress_tensor,
        )
        
        self.set_strain_tensor(strain_tensor)
        
        return self.strain_tensor

    def compute_stress(self) -> None:
        if self.stress_tensor is not None:
            raise ValueError("Stress tensor is already set/computed.")
        
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
                "Strain tensor is not set/computed"
            )
        if not isinstance(self.strain_tensor, SymbolicStrainTensor):
            raise TypeError(
                f"Expected SymbolicStrainTensor, but got {type(self.strain_tensor).__name__}"
            )
            
        stress_tensor = hookes_law_inverse(
            self.material.stiffness_tensor(),
            self.strain_tensor,
        )
        
        self.set_stress_tensor(stress_tensor)
        
        return self.stress_tensor

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
