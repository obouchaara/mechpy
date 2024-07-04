import unittest
import numpy as np
import sympy as sp

from mechpy.core.symbolic.coord import (
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)
from mechpy.core.symbolic.field import (
    SymbolicScalarField,
    SymbolicVectorField,
)
from mechpy.core.symbolic.operators import grad, div, laplacian


class TestSymbolicOperators(unittest.TestCase):
    def test_grad_cartesian(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        f = sp.Function("f")(x, y, z)
        data = sp.NDimArray([f])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = grad(scalar_field)
        expected_data = sp.NDimArray([sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)])
        self.assertEqual(result.data, expected_data)

    def test_grad_cylindrical(self):
        coord_system = SymbolicCylindricalCoordSystem()
        r, theta, z = coord_system.basis
        f = sp.Function("f")(r, theta, z)
        data = sp.NDimArray([f])
        scalar_field = SymbolicScalarField(coord_system, data)

        result = grad(scalar_field)
        expected_data = sp.NDimArray(
            [sp.diff(f, r), sp.diff(f, theta) / r, sp.diff(f, z)]
        )
        self.assertEqual(result.data, expected_data)

    def test_grad_spherical(self):
        coord_system = SymbolicSphericalCoordSystem()
        r, theta, phi = coord_system.basis
        f = sp.Function("f")(r, theta, phi)
        data = sp.NDimArray([f])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = grad(scalar_field)
        expected_data = sp.NDimArray(
            [
                sp.diff(f, r),
                sp.diff(f, theta) / r,
                sp.diff(f, phi) / (r * sp.sin(theta)),
            ]
        )
        self.assertEqual(result.data, expected_data)

    def test_div_cartesian(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        u = sp.Function("u")(x, y, z)
        v = sp.Function("v")(x, y, z)
        w = sp.Function("w")(x, y, z)
        data = sp.NDimArray([u, v, w])
        vector_field = SymbolicVectorField(coord_system, data)
        result = div(vector_field)
        expected_data = sp.NDimArray([sp.diff(u, x) + sp.diff(v, y) + sp.diff(w, z)])
        self.assertEqual(result.data, expected_data)

    def test_div_cylindrical(self):
        coord_system = SymbolicCylindricalCoordSystem()
        r, theta, z = coord_system.basis
        u = sp.Function("u")(r, theta, z)
        v = sp.Function("v")(r, theta, z)
        w = sp.Function("w")(r, theta, z)
        data = sp.NDimArray([u, v, w])
        vector_field = SymbolicVectorField(coord_system, data)
        result = div(vector_field)
        expected_data = sp.NDimArray(
            [sp.diff(u, r) + u / r + sp.diff(v, theta) / r + sp.diff(w, z)]
        )
        self.assertEqual(result.data, expected_data)

    def test_div_spherical(self):
        coord_system = SymbolicSphericalCoordSystem()
        r, theta, phi = coord_system.basis
        u = sp.Function("u")(r, theta, phi)
        v = sp.Function("v")(r, theta, phi)
        w = sp.Function("w")(r, theta, phi)
        data = sp.NDimArray([u, v, w])
        vector_field = SymbolicVectorField(coord_system, data)
        result = div(vector_field)
        expected_data = sp.NDimArray(
            [
                sp.diff(u, r)
                + 2 * u / r
                + sp.diff(v, theta) / r
                + sp.cot(theta) * v / r
                + sp.diff(w, phi) / (r * sp.sin(theta))
            ]
        )
        self.assertEqual(result.data, expected_data)

    def test_laplacian_scalar_cartesian(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        f = sp.Function("f")(x, y, z)
        data = sp.NDimArray([f])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = laplacian(scalar_field)
        expected_data = sp.NDimArray(
            [sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)]
        )
        self.assertEqual(result.data, expected_data)

    def test_laplacian_vector_cartesian(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        u = sp.Function("u")(x, y, z)
        v = sp.Function("v")(x, y, z)
        w = sp.Function("w")(x, y, z)
        data = sp.NDimArray([u, v, w])
        vector_field = SymbolicVectorField(coord_system, data)
        result = laplacian(vector_field)
        expected_data = sp.NDimArray(
            [
                sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2),
                sp.diff(v, x, 2) + sp.diff(v, y, 2) + sp.diff(v, z, 2),
                sp.diff(w, x, 2) + sp.diff(w, y, 2) + sp.diff(w, z, 2),
            ]
        )
        self.assertEqual(result.data, expected_data)

    def test_laplacian_scalar_cylindrical(self):
        coord_system = SymbolicCylindricalCoordSystem()
        r, theta, z = coord_system.basis
        f = sp.Function("f")(r, theta, z)
        data = sp.NDimArray([f])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = laplacian(scalar_field)
        expected_data = sp.NDimArray(
            [
                sp.diff(f, r, 2)
                + sp.diff(f, r) / r
                + sp.diff(f, theta, 2) / r**2
                + sp.diff(f, z, 2)
            ]
        )
        self.assertEqual(result.data, expected_data)

    def test_laplacian_scalar_spherical(self):
        coord_system = SymbolicSphericalCoordSystem()
        r, theta, phi = coord_system.basis
        f = sp.Function("f")(r, theta, phi)
        data = sp.NDimArray([f])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = laplacian(scalar_field)
        expected_data = sp.NDimArray(
            [
                sp.diff(f, r, 2)
                + (2 / r) * sp.diff(f, r)
                + sp.diff(f, theta, 2) / r**2
                + sp.cot(theta) * sp.diff(f, theta) / r**2
                + sp.diff(f, phi, 2) / (r**2 * sp.sin(theta) ** 2)
            ]
        )
        self.assertEqual(result.data, expected_data)


class TestSymbolicOperatorsSpecificFields(unittest.TestCase):
    def test_div_specific_field_1(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        data = sp.NDimArray([x**2, y**2, z**2])
        vector_field = SymbolicVectorField(coord_system, data)
        result = div(vector_field)
        expected_data = sp.NDimArray([2*x + 2*y + 2*z])
        self.assertEqual(result.data, expected_data)

    def test_div_specific_field_2(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        data = sp.NDimArray([x*y, y*z, z*x])
        vector_field = SymbolicVectorField(coord_system, data)
        result = div(vector_field)
        expected_data = sp.NDimArray([y + z + x])
        self.assertEqual(result.data, expected_data)

    def test_grad_specific_field_1(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        data = sp.NDimArray([x**2 + y**2 + z**2])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = grad(scalar_field)
        expected_data = sp.NDimArray([2*x, 2*y, 2*z])
        self.assertEqual(result.data, expected_data)

    def test_grad_specific_field_2(self):
        coord_system = SymbolicSphericalCoordSystem()
        r, theta, phi = coord_system.basis
        data = sp.NDimArray([r**2 * sp.sin(theta)])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = grad(scalar_field)
        expected_data = sp.NDimArray([2*r*sp.sin(theta), r*sp.cos(theta), 0])
        self.assertEqual(result.data, expected_data)

    def test_laplacian_specific_field_1(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        data = sp.NDimArray([x**2 + y**2 + z**2])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = laplacian(scalar_field)
        expected_data = sp.NDimArray([6])
        self.assertEqual(result.data, expected_data)

    def test_laplacian_specific_field_2(self):
        coord_system = SymbolicCylindricalCoordSystem()
        r, theta, z = coord_system.basis
        data = sp.NDimArray([r**2 * sp.cos(theta)])
        scalar_field = SymbolicScalarField(coord_system, data)
        result = laplacian(scalar_field)
        expected_data = sp.NDimArray([3 * sp.cos(theta)])
        self.assertEqual(result.data, expected_data)

    def test_div_curl_identity(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        data = sp.NDimArray([y*z, z*x, x*y])
        vector_field = SymbolicVectorField(coord_system, data)
        curl_x = sp.diff(vector_field.data[2], y) - sp.diff(vector_field.data[1], z)
        curl_y = sp.diff(vector_field.data[0], z) - sp.diff(vector_field.data[2], x)
        curl_z = sp.diff(vector_field.data[1], x) - sp.diff(vector_field.data[0], y)
        curl_field = SymbolicVectorField(coord_system, sp.NDimArray([curl_x, curl_y, curl_z]))
        result = div(curl_field)
        expected_data = sp.NDimArray([0])
        self.assertEqual(result.data, expected_data)

    def test_curl_grad_identity(self):
        coord_system = SymbolicCartesianCoordSystem()
        x, y, z = coord_system.basis
        data = sp.NDimArray([x**2*y + y**2*z + z**2*x])
        scalar_field = SymbolicScalarField(coord_system, data)
        grad_field = grad(scalar_field)
        curl_x = sp.diff(grad_field.data[2], y) - sp.diff(grad_field.data[1], z)
        curl_y = sp.diff(grad_field.data[0], z) - sp.diff(grad_field.data[2], x)
        curl_z = sp.diff(grad_field.data[1], x) - sp.diff(grad_field.data[0], y)
        result = sp.NDimArray([curl_x, curl_y, curl_z])
        expected_data = sp.NDimArray([0, 0, 0])
        self.assertEqual(result, expected_data)

if __name__ == "__main__":
    unittest.main()
