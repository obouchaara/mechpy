import sympy as sp

from .coord import (
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)
from .field import (
    SymbolicSpatialField,
    SymbolicScalarField,
    SymbolicVectorField,
)


def grad(scalar_field: SymbolicScalarField) -> SymbolicVectorField:
    """
    Calculate the gradient of a scalar field.
    """
    coord_system = scalar_field.coord_system
    f = scalar_field.data[0]

    if isinstance(coord_system, SymbolicCartesianCoordSystem):
        x, y, z = coord_system.basis
        data = sp.NDimArray([sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)])
    elif isinstance(coord_system, SymbolicCylindricalCoordSystem):
        r, theta, z = coord_system.basis
        data = sp.NDimArray([sp.diff(f, r), sp.diff(f, theta) / r, sp.diff(f, z)])
    elif isinstance(coord_system, SymbolicSphericalCoordSystem):
        r, theta, phi = coord_system.basis
        data = sp.NDimArray(
            [
                sp.diff(f, r),
                sp.diff(f, theta) / r,
                sp.diff(f, phi) / (r * sp.sin(theta)),
            ]
        )
    else:
        raise NotImplementedError

    return SymbolicVectorField(coord_system, data)


def div(vector_field: SymbolicVectorField) -> SymbolicScalarField:
    """
    Calculate the divergence of a vector field.
    """
    coord_system = vector_field.coord_system
    u, v, w = vector_field.data

    if isinstance(coord_system, SymbolicCartesianCoordSystem):
        x, y, z = coord_system.basis
        data = sp.NDimArray([sp.diff(u, x) + sp.diff(v, y) + sp.diff(w, z)])
    elif isinstance(coord_system, SymbolicCylindricalCoordSystem):
        r, theta, z = coord_system.basis
        data = sp.NDimArray(
            [sp.diff(u, r) + u / r + sp.diff(v, theta) / r + sp.diff(w, z)]
        )
    elif isinstance(coord_system, SymbolicSphericalCoordSystem):
        r, theta, phi = coord_system.basis
        data = sp.NDimArray(
            [
                sp.diff(u, r)
                + 2 * u / r
                + sp.diff(v, theta) / r
                + sp.cot(theta) * v / r
                + sp.diff(w, phi) / (r * sp.sin(theta))
            ]
        )
    else:
        raise NotImplementedError

    return SymbolicScalarField(coord_system, data)


def laplacian(field: SymbolicSpatialField) -> SymbolicSpatialField:
    """
    Calculate the Laplacian of a field.
    """
    if isinstance(field, SymbolicScalarField):
        return div(grad(field))
    elif isinstance(field, SymbolicVectorField):
        return vector_laplacian(field)
    else:
        raise NotImplementedError


def vector_laplacian(vector_field: SymbolicVectorField) -> SymbolicVectorField:
    """
    Calculate the vector Laplacian (∇²A) of a vector field.
    """
    coord_system = vector_field.coord_system
    u, v, w = vector_field.data

    if isinstance(coord_system, SymbolicCartesianCoordSystem):
        x, y, z = coord_system.basis
        data = sp.NDimArray(
            [
                sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2),
                sp.diff(v, x, 2) + sp.diff(v, y, 2) + sp.diff(v, z, 2),
                sp.diff(w, x, 2) + sp.diff(w, y, 2) + sp.diff(w, z, 2),
            ]
        )
    elif isinstance(coord_system, SymbolicCylindricalCoordSystem):
        r, theta, z = coord_system.basis
        data = sp.NDimArray(
            [
                sp.diff(u, r, 2)
                + sp.diff(u, r) / r
                + sp.diff(u, theta, 2) / r**2
                + sp.diff(u, z, 2)
                - u / r**2
                - 2 * sp.diff(v, theta) / r**2,
                sp.diff(v, r, 2)
                + sp.diff(v, r) / r
                + sp.diff(v, theta, 2) / r**2
                + sp.diff(v, z, 2)
                - v / r**2
                + 2 * sp.diff(u, theta) / r**2,
                sp.diff(w, r, 2)
                + sp.diff(w, r) / r
                + sp.diff(w, theta, 2) / r**2
                + sp.diff(w, z, 2),
            ]
        )
    elif isinstance(coord_system, SymbolicSphericalCoordSystem):
        r, theta, phi = coord_system.basis
        data = sp.NDimArray(
            [
                sp.diff(u, r, 2)
                + 2 * sp.diff(u, r) / r
                - 2 * u / r**2
                + sp.diff(u, theta, 2) / r**2
                + sp.cot(theta) * sp.diff(u, theta) / r**2
                + sp.diff(u, phi, 2) / (r**2 * sp.sin(theta) ** 2)
                - 2 * sp.diff(v, theta) / r**2
                - 2 * sp.diff(w, phi) / (r**2 * sp.sin(theta)),
                sp.diff(v, r, 2)
                + 2 * sp.diff(v, r) / r
                + sp.diff(v, theta, 2) / r**2
                + sp.cot(theta) * sp.diff(v, theta) / r**2
                + sp.diff(v, phi, 2) / (r**2 * sp.sin(theta) ** 2)
                - v / (r**2 * sp.sin(theta) ** 2)
                + 2 * sp.diff(u, theta) / r**2
                - 2 * sp.cot(theta) * sp.diff(w, phi) / (r**2 * sp.sin(theta)),
                sp.diff(w, r, 2)
                + 2 * sp.diff(w, r) / r
                + sp.diff(w, theta, 2) / r**2
                + sp.cot(theta) * sp.diff(w, theta) / r**2
                + sp.diff(w, phi, 2) / (r**2 * sp.sin(theta) ** 2)
                - w / (r**2 * sp.sin(theta) ** 2)
                + 2 * sp.diff(u, phi) / (r**2 * sp.sin(theta))
                + 2 * sp.cot(theta) * sp.diff(v, phi) / (r**2 * sp.sin(theta)),
            ]
        )
    else:
        raise NotImplementedError

    return SymbolicVectorField(coord_system, data)
