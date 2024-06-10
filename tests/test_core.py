import unittest
import numpy as np
import sympy as sp

from mechpy.core.symbolic.coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)

from mechpy.core.symbolic.field import (
    SymbolicField,
    SymbolicSpatialField,
    SymbolicScalarField,
)

from mechpy.core.symbolic.tensor import (
    SymbolicThreeByThreeTensor,
    SymbolicSymmetricThreeByThreeTensor,
)


class TestSymbolicCoordSystem(unittest.TestCase):
    def test_init(self):
        origin = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis = sp.symbols("x1 x2 x3")
        coord_system = SymbolicCoordSystem(origin=origin, basis=basis)
        self.assertEqual(coord_system.origin, origin)
        self.assertEqual(coord_system.basis, basis)

        origin = (sp.Float(0), sp.Float(0))
        basis = sp.symbols("x1 x2")
        coord_system = SymbolicCoordSystem(origin=origin, basis=basis)
        self.assertEqual(coord_system.origin, origin)
        self.assertEqual(coord_system.basis, basis)

        with self.assertRaises(ValueError) as context:
            origin = (0, 0)
            basis = sp.symbols("x1 x2")
            SymbolicCoordSystem(origin=origin, basis=basis)
        self.assertEqual(
            str(context.exception), "origin must be a tuple of sympy.Number"
        )

    def test_auto_detect(self):
        with self.assertRaises(NotImplementedError) as context:
            SymbolicCoordSystem.auto_detect(data=None)

    def test_coord_subs(self):
        basis = sp.symbols("x1 x2 x3")
        x1, x2, x3 = basis
        y1, y2, y3 = sp.symbols("y1 y2 y3")
        subs = {
            x1: y3,
            x2: y1,
            x3: y2,
        }
        new_coord = SymbolicCoordSystem.coord_subs(basis, subs)
        self.assertEqual(new_coord, (y3, y1, y2))
        subs = {
            x1: y1 + 2 * y2,
            x2: 5 * y2 - 2 * y3,
            x3: y2 - 2 * y1,
        }
        new_coord = SymbolicCoordSystem.coord_subs(basis, subs)
        subs = {
            y1: x1,
            y2: 5 * x2 - 2 * x1,
            y3: x2 - 2 * x3,
        }
        new_coord = SymbolicCoordSystem.coord_subs(new_coord, subs)
        self.assertEqual(
            new_coord,
            (
                -3 * x1 + 10 * x2,
                -10 * x1 + 23 * x2 + 4 * x3,
                -4 * x1 + 5 * x2,
            ),
        )

    def test_coord_eval(self):
        x1, x2, x3 = sp.symbols("x1 x2 x3")
        basis = (-3 * x1 + 10 * x2, -10 * x1 + 23 * x2 + 4 * x3, -4 * x1 + 5 * x2)
        coord_val = SymbolicCoordSystem.coord_eval(basis, subs={x1: 1, x2: 2, x3: 3})
        self.assertEqual(coord_val, (17, 48, 6))


class TestSymbolicCartesianCoordSystem(unittest.TestCase):
    def test_init(self):
        origin = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis = sp.symbols("x y z")
        cartesian_system = SymbolicCartesianCoordSystem(origin=origin, basis=basis)
        self.assertEqual(cartesian_system.origin, origin)
        self.assertEqual(cartesian_system.basis, basis)

        origin = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis = sp.symbols("x y z")
        cartesian_system = SymbolicCartesianCoordSystem()
        self.assertEqual(cartesian_system.origin, origin)
        self.assertEqual(cartesian_system.basis, basis)

        origin = (sp.Float(0), sp.Float(0))
        basis = sp.symbols("x y")
        with self.assertRaises(ValueError) as context:
            SymbolicCartesianCoordSystem(origin=origin, basis=basis)
        self.assertEqual(str(context.exception), "basis must have a length of 3")

    def test_get_basis_cylindrical_exprs(self):
        cartesian_system = SymbolicCartesianCoordSystem()
        x, y, z = cartesian_system.basis
        r, theta, z = sp.symbols("r theta z")
        cylindrical_exprs = cartesian_system.get_basis_cylindrical_exprs((r, theta, z))
        self.assertEqual(cylindrical_exprs[r], sp.sqrt(x**2 + y**2))
        self.assertEqual(cylindrical_exprs[theta], sp.atan2(y, x))
        self.assertEqual(cylindrical_exprs[z], z)

    def test_get_basis_spherical_exprs(self):
        cartesian_system = SymbolicCartesianCoordSystem()
        x, y, z = cartesian_system.basis
        r, theta, phi = sp.symbols("r theta phi")
        spherical_exprs = cartesian_system.get_basis_spherical_exprs((r, theta, phi))
        self.assertEqual(spherical_exprs[r], sp.sqrt(x**2 + y**2 + z**2))
        self.assertEqual(spherical_exprs[theta], sp.atan2(y, x))
        self.assertEqual(spherical_exprs[phi], sp.acos(z / sp.sqrt(x**2 + y**2 + z**2)))

    def test_to_cylindrical(self):
        pass

    def test_to_spherical(self):
        pass

    def test_get_cylindrical_coord(self):
        a, b = sp.symbols("a b", positive=True)
        cartesian_coord = (0, a, b)
        symbolic_cartesian_system = SymbolicCartesianCoordSystem()
        cylindrical_coord = symbolic_cartesian_system.get_cylindrical_coord(
            cartesian_coord
        )
        self.assertEqual(cylindrical_coord, (a, sp.pi / 2, b))

        eval_coord = SymbolicCoordSystem.coord_eval(
            cylindrical_coord, subs={a: 3, b: 5}
        )
        self.assertEqual(eval_coord[0], 3)
        self.assertAlmostEqual(eval_coord[1], 1.5707963267949)
        self.assertEqual(eval_coord[2], 5)

    def test_get_spherical_coord(self):
        a = sp.symbols("a", positive=True)
        cartesian_coord = (0, a, 0)
        symbolic_cartesian_system = SymbolicCartesianCoordSystem()
        spherical_coord = symbolic_cartesian_system.get_spherical_coord(cartesian_coord)
        self.assertEqual(spherical_coord, (a, sp.pi / 2, sp.pi / 2))

        eval_coord = SymbolicCoordSystem.coord_eval(spherical_coord, subs={a: 1})
        self.assertEqual(eval_coord[0], 1)
        self.assertAlmostEqual(eval_coord[1], 1.5707963267949)
        self.assertAlmostEqual(eval_coord[2], 1.5707963267949)


class TestSymbolicCylindricalCoordSystem(unittest.TestCase):
    def test_init(self):
        origin = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis = sp.symbols("r theta z")
        cylindrical_system = SymbolicCylindricalCoordSystem(origin=origin, basis=basis)
        self.assertEqual(cylindrical_system.origin, origin)
        self.assertEqual(cylindrical_system.basis, basis)

        origin = (sp.Float(0), sp.Float(0), sp.Float(0))
        basis = sp.symbols("r theta z")
        cylindrical_system = SymbolicCylindricalCoordSystem()
        self.assertEqual(cylindrical_system.origin, origin)
        self.assertEqual(cylindrical_system.basis, basis)

    def test_get_basis_cartesian_exprs(self):
        cylindrical_system = SymbolicCylindricalCoordSystem()
        r, theta, z = cylindrical_system.basis
        x, y, z = sp.symbols("x y z")
        cartesian_exprs = cylindrical_system.get_basis_cartesian_exprs((x, y, z))
        self.assertEqual(cartesian_exprs[x], r * sp.cos(theta))
        self.assertEqual(cartesian_exprs[y], r * sp.sin(theta))
        self.assertEqual(cartesian_exprs[z], z)

    def test_to_cartesian(self):
        pass

    def test_get_cartesian_coords(self):
        a, b = sp.symbols("a b", positive=True)
        cylindrical_coord = (2 * a, sp.pi / 3, b)
        cylindrical_system = SymbolicCylindricalCoordSystem()
        cartesian_coord = cylindrical_system.get_cartesian_coords(cylindrical_coord)
        self.assertEqual(cartesian_coord, (a, sp.sqrt(3) * a, b))

        eval_coord = SymbolicCoordSystem.coord_eval(cartesian_coord, subs={a: 1, b: 5})
        self.assertEqual(eval_coord[0], 1)
        self.assertAlmostEqual(eval_coord[1], 1.73205080756888)
        self.assertEqual(eval_coord[2], 5)


class TestSymbolicField(unittest.TestCase):
    def test_init(self):
        coord_system = SymbolicCartesianCoordSystem()
        data = sp.Array([1, 2, 3])
        field = SymbolicField(coord_system, data)
        self.assertEqual(field.coord_system, coord_system)
        self.assertEqual(field.data, data)
        self.assertEqual(field.field_params, {})

        coord_system = SymbolicCartesianCoordSystem()
        data = sp.Array([1, 2, 3])
        n, m = sp.symbols("n m")
        field_params = {
            n: None,
            m: None,
        }
        field = SymbolicField(coord_system, data, field_params)
        self.assertEqual(field.coord_system, coord_system)
        self.assertEqual(field.data, data)
        self.assertEqual(field.field_params, field_params)

        coord_system = SymbolicCartesianCoordSystem()
        n, m = sp.symbols("n m")
        data = sp.Array([1 + n, 2 + m, 3])
        field_params = {n, m}
        with self.assertRaises(ValueError) as context:
            SymbolicField(coord_system, data, field_params)
        self.assertEqual(
            str(context.exception),
            "Field parameters must be a dict.",
        )

        coord_system = SymbolicCartesianCoordSystem()
        x1, x2, x3 = coord_system.basis
        data = sp.Array([1, 2, 3])
        field_params = {
            x1: None,
        }
        with self.assertRaises(ValueError) as context:
            SymbolicField(coord_system, data, field_params)
        self.assertEqual(
            str(context.exception),
            "Field parameters must not overlap with coordinate system basis symbols.",
        )

        coord_system = SymbolicCartesianCoordSystem()
        x1, x2, x3 = coord_system.basis
        n, m = sp.symbols("n m")
        data = sp.Array([1 + n + x1, 2 + m, 3 + x2])
        field_params = {
            n: None,
        }
        with self.assertRaises(ValueError) as context:
            SymbolicField(coord_system, data, field_params)
        self.assertEqual(
            str(context.exception),
            f"The field data contains symbols not in the basis or field parameters: {m}",
        )

    def test_get_invalid_symbols(self):
        pass

    def test_subs_field_params(self):
        coord_system = SymbolicCartesianCoordSystem()
        x1, x2, x3 = coord_system.basis
        n, m, p = sp.symbols("n m p")
        data = sp.Array([(1 + 3 * n) * x1, 2 + m, 3])
        field_params = {
            n: None,
            m: None,
        }
        field = SymbolicField(coord_system, data, field_params)

        field.subs_field_params({n: 1})
        self.assertEqual(field.data, sp.Array([4 * x1, 2 + m, 3]))
        self.assertEqual(field.field_params, {m: None})

        field.subs_field_params({m: 3})
        self.assertEqual(field.data, sp.Array([4 * x1, 5, 3]))
        self.assertEqual(field.field_params, {})

        with self.assertRaises(ValueError) as context:
            field.subs_field_params({p: 5})
        self.assertEqual(
            str(context.exception),
            f"Parameter {p} not found in field parameters",
        )

        coord_system = SymbolicCartesianCoordSystem()
        a, b, c = coord_system.basis
        n, m, p = sp.symbols("n m p")
        data = sp.Array([(1 + 3 * n) * x1, 2 + m, 3])
        field_params = {
            n: None,
            m: None,
            a: None,
        }
        with self.assertRaises(ValueError) as context:
            SymbolicField(coord_system, data, field_params)
        self.assertEqual(
            str(context.exception),
            "Field parameters must not overlap with coordinate system basis symbols.",
        )

    def test_to_cartesian(self):
        coord_system = SymbolicCylindricalCoordSystem()
        x1, x2, x3 = coord_system.basis
        data = sp.Array([x1 + x3 * x3])
        scalar_field = SymbolicScalarField.create(coord_system, data)
        new_scalar_field = scalar_field.to_cartesian()
        new_basis = new_scalar_field.coord_system.basis
        y1, y2, y3 = new_basis
        new_data = new_scalar_field.data
        self.assertEqual(new_data, sp.NDimArray([y3**2 + sp.sqrt(y1**2 + y2**2)]))

    def test_cylindrical(self):
        pass

    def test_to_spherical(self):
        pass

    def test_subs(self):
        pass


class TestSymbolicSpatialField(unittest.TestCase):
    def test_init(self):
        pass


class TestSymbolicScalarField(unittest.TestCase):
    def test_create(self):
        field = SymbolicScalarField.create()
        self.assertIsInstance(field.coord_system, SymbolicCartesianCoordSystem)
        x1, x2, x3 = field.coord_system.basis
        self.assertEqual(field.data, sp.NDimArray([sp.Function("f")(x1, x2, x3)]))

        coord_system = SymbolicCartesianCoordSystem()
        x1, x2, x3 = coord_system.basis
        self.assertEqual(field.coord_system.basis, (x1, x2, x3))
        data = sp.NDimArray([x1 * x1 - x2 * x2])
        field = SymbolicScalarField.create(coord_system=coord_system, data=data)
        self.assertEqual(field.data, sp.NDimArray([x1 * x1 - x2 * x2]))


# class TestSymbolicThreeByThreeTensor(unittest.TestCase):
#     def test_initialization_valid_data(self):
#         valid_array = sp.ImmutableDenseNDimArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#         tensor = SymbolicThreeByThreeTensor(valid_array)
#         self.assertEqual(tensor.data, valid_array)

#     def test_initialization_invalid_data(self):
#         invalid_matrix = sp.ImmutableDenseNDimArray([[1, 2], [3, 4]])
#         with self.assertRaises(ValueError):
#             SymbolicThreeByThreeTensor(invalid_matrix)

#     def test_to_symmetric_valid(self):
#         symmetric_array = sp.ImmutableDenseNDimArray([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
#         tensor = SymbolicThreeByThreeTensor(symmetric_array)
#         symmetric_tensor = tensor.to_symmetric()
#         self.assertIsInstance(symmetric_tensor, SymbolicSymmetricThreeByThreeTensor)
#         self.assertTrue(symmetric_tensor.is_symmetric())

#     def test_to_symmetric_invalid(self):
#         non_symmetric_array = sp.ImmutableDenseNDimArray(
#             [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#         )
#         tensor = SymbolicThreeByThreeTensor(non_symmetric_array)
#         with self.assertRaises(ValueError):
#             tensor.to_symmetric()


# class TestSymbolicSymmetricThreeByThreeTensor(unittest.TestCase):
#     def test_notation_standard_map(self):
#         expected_maps = {
#             2: {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2},
#             3: {
#                 (0, 0): 0,
#                 (0, 1): 1,
#                 (0, 2): 2,
#                 (1, 0): 1,
#                 (1, 1): 3,
#                 (1, 2): 4,
#                 (2, 0): 2,
#                 (2, 1): 4,
#                 (2, 2): 5,
#             },
#             4: {
#                 (0, 0): 0,
#                 (0, 1): 1,
#                 (0, 2): 2,
#                 (0, 3): 3,
#                 (1, 0): 1,
#                 (1, 1): 4,
#                 (1, 2): 5,
#                 (1, 3): 6,
#                 (2, 0): 2,
#                 (2, 1): 5,
#                 (2, 2): 7,
#                 (2, 3): 8,
#                 (3, 0): 3,
#                 (3, 1): 6,
#                 (3, 2): 8,
#                 (3, 3): 9,
#             },
#         }

#         for n, expected_map in expected_maps.items():
#             assert (
#                 SymbolicSymmetricThreeByThreeTensor.notation_standard_map(n)
#                 == expected_map
#             )

#     def test_notation_standard_inverse_map(self):
#         expected_inverse_maps = {
#             2: {0: (0, 0), 1: (0, 1), 2: (1, 1)},
#             3: {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 1), 4: (1, 2), 5: (2, 2)},
#             4: {
#                 0: (0, 0),
#                 1: (0, 1),
#                 2: (0, 2),
#                 3: (0, 3),
#                 4: (1, 1),
#                 5: (1, 2),
#                 6: (1, 3),
#                 7: (2, 2),
#                 8: (2, 3),
#                 9: (3, 3),
#             },
#         }

#         for n, expected_inverse_map in expected_inverse_maps.items():
#             assert (
#                 SymbolicSymmetricThreeByThreeTensor.notation_standard_inverse_map(n)
#                 == expected_inverse_map
#             )


if __name__ == "__main__":
    unittest.main()
