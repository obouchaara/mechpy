import sympy as sp

from .coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
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

        self.coord_system = coord_system
        self.material = material
        self.displacement = displacement
        self.volume_force = volume_force or SymbolicVolumeForce(
            coord_system=self.coord_system,
            data=sp.NDimArray([0, 0, 0]),
        )

    def general_equation(self):
        if not hasattr(self.material, "rho"):
            raise ValueError("the material should include rho for general equation")

        lamda, mu = self.material.get_lame_params()
        rho = self.material.rho
        field = self.displacement
        t = sp.Symbol("t")

        grad_div_field = grad(div(field))
        laplace_field = laplacian(field)

        lhs = rho * sp.diff(field.data, t, t)
        rhs = (lamda + mu) * grad_div_field.data + mu * laplace_field.data
        f = self.volume_force.data

        return lhs - rhs - f

    def static_equation(self):
        lamda, mu = self.material.get_lame_params()
        field = self.displacement

        grad_div_field = grad(div(field))
        laplace_field = laplacian(field)

        rhs = (lamda + mu) * grad_div_field.data + mu * laplace_field.data
        f = self.volume_force.data

        return rhs + f
