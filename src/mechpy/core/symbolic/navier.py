import sympy as sp

from .coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
    SymbolicDynamicCoordSystem,
)
from .displacement import SymbolicDisplacement
from .material import SymbolicIsotropicMaterial
from .solicitation import SymbolicVolumeForce
from .operators import grad, div, laplacian


class SymbolicNavier:
    def __init__(
        self,
        coord_system: SymbolicCoordSystem,
        material: SymbolicIsotropicMaterial,
        displacement: SymbolicDisplacement,
        volume_force: SymbolicVolumeForce = None,
    ):

        ## check compatibility of all coord system
        # to implement

        if isinstance(coord_system, SymbolicDynamicCoordSystem) and not hasattr(
            material, "rho"
        ):
            raise ValueError("the material should include rho for dynamic.")

        self.coord_system = coord_system
        self.material = material
        self.displacement = displacement
        self.volume_force = volume_force or SymbolicVolumeForce(
            coord_system=self.coord_system,
            data=sp.NDimArray([0, 0, 0]),
        )

    def static_equation(self):
        lamda, mu = self.material.get_lame_params()

        field = self.displacement
        grad_div_field = grad(div(field))
        laplace_field = laplacian(field)

        rhs = (lamda + mu) * grad_div_field.data + mu * laplace_field.data
        f = self.volume_force.data

        return rhs + f

    def general_equation(self):
        if not isinstance(self.coord_system, SymbolicDynamicCoordSystem):
            raise ValueError("the coord system should be dynamic for general equation.")
        
        lamda, mu = self.material.get_lame_params()
        
        field = self.displacement
        grad_div_field = grad(div(field))
        laplace_field = laplacian(field)
        
        f = self.volume_force.data
        rhs = (lamda + mu) * grad_div_field.data + mu * laplace_field.data
        
        if isinstance(self.coord_system, SymbolicDynamicCoordSystem):
            t = self.coord_system.time_symbol
            rho = self.material.rho

            lhs = rho * sp.diff(field.data, t, t)

            return lhs - rhs - f

        return rhs + f
