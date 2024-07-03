import sympy as sp

from .coord import (
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)
from .field import SymbolicVectorField
from .strain import SymbolicStrainTensor


class SymbolicDisplacement(SymbolicVectorField):
    def __init__(self, coord_system, data, field_params=None, symbols_validation=True):
        super().__init__(coord_system, data, field_params, symbols_validation)

    def __repr__(self):
        return f"SymbolicDisplacement(\n{self.data}\n)"

    def strain_tensor(self, coord_system=None) -> SymbolicStrainTensor:
        if coord_system is None:
            coord_system = self.coord_system

        if isinstance(coord_system, SymbolicCartesianCoordSystem):
            x, y, z = coord_system.basis
            u, v, w = self.data
            e_11 = sp.diff(u, x)
            e_22 = sp.diff(v, y)
            e_33 = sp.diff(w, z)
            e_12 = sp.diff(u, y) + sp.diff(v, x)
            e_23 = sp.diff(v, z) + sp.diff(w, y)
            e_31 = sp.diff(w, x) + sp.diff(u, z)

        elif isinstance(coord_system, SymbolicCylindricalCoordSystem):
            r, theta, z = coord_system.basis
            u, v, w = self.data
            e_11 = sp.diff(u, r)
            e_22 = (1 / r) * sp.diff(v, theta) + u / r
            e_33 = sp.diff(w, z)
            e_12 = r * sp.diff((1 / r) * v, r) + (1 / r) * sp.diff(u, theta)
            e_23 = (1 / r) * sp.diff(w, theta) + sp.diff(v, z)
            e_31 = sp.diff(w, r) + sp.diff(u, z)

        elif isinstance(coord_system, SymbolicSphericalCoordSystem):
            r, theta, phi = coord_system.basis
            u, v, w = self.data
            e_11 = sp.diff(u, r)
            e_22 = (1 / r) * sp.diff(v, theta) + u / r
            e_33 = (
                (1 / r) * sp.diff(w, phi)
                + (1 / (r * sp.sin(theta))) * sp.diff(v, phi)
                + u / r
            )
            e_12 = r * sp.diff((1 / r) * v, r) + (1 / r) * sp.diff(u, theta) - v / r
            e_23 = (
                (1 / r) * sp.diff(w, theta)
                + (1 / (r * sp.sin(theta))) * sp.diff(v, phi)
                - w / (r * sp.tan(theta))
            )
            e_31 = r * sp.diff((1 / r) * w, r) + (1 / (r * sp.sin(theta))) * sp.diff(
                u, phi
            )

        else:
            raise ValueError("Unsupported coordinate system")

        components = [e_11, e_22, e_33, e_12, e_23, e_31]
        strain_tensor = SymbolicStrainTensor.from_list(components, notation="voigt")
        return strain_tensor
