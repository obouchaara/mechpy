import sympy as sp

from .coord import (
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)
from .displacement import SymbolicDisplacement
from .material import SymbolicIsotropicMaterial
from .operators import grad, div, laplacian


class SymbolicNavier:
    def __init__(
        self,
        material: SymbolicIsotropicMaterial,
        displacement: SymbolicDisplacement,
        # solicitation: SymbolicSolicitation = None # to implement
    ):

        self.material = material
        self.displacement = displacement
        # self.solicitation = solicitation # to implement

    def general_equation(self):
        if not hasattr(self.material, "rho"):
            raise ValueError("the material should include rho for general equation")

        lamda, mu = self.material.get_lame_params()
        rho = self.material.rho
        field = self.displacement
        f = sp.NDimArray([sp.Symbol(f"f_{_}") for _ in field.coord_system.basis])
        t = sp.Symbol("t")

        grad_div_field = grad(div(field))
        laplace_field = laplacian(field)

        lhs = rho * sp.diff(field.data, t, t)
        rhs = (lamda + mu) * grad_div_field.data + mu * laplace_field.data

        return lhs - rhs - f

    def static_equation(self):
        lamda, mu = self.material.get_lame_params()
        field = self.displacement
        f = sp.NDimArray([sp.Symbol(f"f_{_}") for _ in field.coord_system.basis])

        grad_div_field = grad(div(field))
        laplace_field = laplacian(field)

        rhs = (lamda + mu) * grad_div_field.data + mu * laplace_field.data

        return rhs + f
