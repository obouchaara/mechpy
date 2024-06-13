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
    SymbolicTensor,
    SymbolicThreeByThreeTensor,
    SymbolicSixBySixTensor,
    SymbolicSymmetricThreeByThreeTensor,
)

from mechpy.core.symbolic.stress import (
    SymbolicStressTensor,
)

from mechpy.core.symbolic.strain import (
    SymbolicStrainTensor,
)

from mechpy.core.symbolic.displacement import (
    SymbolicDisplacement,
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

    def test_plot(self):
        n, m = sp.symbols("n m")
        field_params = {
            n: {-0.2, -0.1, 0, 0.1, 0.2},
            m: {-0.2, -0.1, 0, 0.1, 0.2},
        }
        data = sp.Array([1 * n, -2 * m, 0])
        linear_scalar_field = SymbolicScalarField.create_linear(
            data=data,
            field_params=field_params,
        )
        try:
            linear_scalar_field.plot()
        except Exception as e:
            self.fail(f"Test failed due to unexpected exception: {e}")

        n, m = sp.symbols("n m")
        field_params = {
            n: {0, 0.1, 0.2, 0.3, 0.4, 0.5},
            m: None,
        }
        data = sp.Array([1 * n, -2 * m, 0])
        linear_scalar_field = SymbolicScalarField.create_linear(
            data=data,
            field_params=field_params,
        )
        with self.assertRaises(ValueError) as context:
            linear_scalar_field.plot()
        self.assertEqual(
            str(context.exception),
            "the param m values in not defined",
        )


class TestSymbolicTensor(unittest.TestCase):
    def test_init(self):
        data = sp.Array([1, 2, 3])
        tensor = SymbolicTensor(data)
        self.assertEqual(tensor.data, data)

        with self.assertRaises(ValueError) as context:
            SymbolicTensor(data=None)
        self.assertEqual(
            str(context.exception),
            "Input data must be a SymPy NDimArray.",
        )

    def test_notation_standard_map(self):
        index_map = SymbolicTensor.notation_standard_map(1)
        self.assertEqual(index_map, {(0, 0): 0})

        index_map = SymbolicTensor.notation_standard_map(2)
        self.assertEqual(index_map, {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2})

        index_map = SymbolicTensor.notation_standard_map(3)
        self.assertEqual(
            index_map,
            {
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 1,
                (0, 2): 2,
                (2, 0): 2,
                (1, 1): 3,
                (1, 2): 4,
                (2, 1): 4,
                (2, 2): 5,
            },
        )

    def test_notation_standard_inverse_map(self):
        index_map_inverse = SymbolicTensor.notation_standard_inverse_map(1)
        self.assertEqual(index_map_inverse, {0: (0, 0)})

        index_map_inverse = SymbolicTensor.notation_standard_inverse_map(2)
        self.assertEqual(index_map_inverse, {0: (0, 0), 1: (0, 1), 2: (1, 1)})

        index_map_inverse = SymbolicTensor.notation_standard_inverse_map(3)
        self.assertEqual(
            index_map_inverse,
            {
                0: (0, 0),
                1: (0, 1),
                2: (0, 2),
                3: (1, 1),
                4: (1, 2),
                5: (2, 2),
            },
        )

    def test_rank(self):
        data = sp.Array([[1, 2], [3, 4]])
        tensor = SymbolicTensor(data)
        self.assertTrue(tensor.is_second_rank())
        self.assertFalse(tensor.is_fourth_rank())

        data = sp.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = SymbolicTensor(data)
        self.assertTrue(tensor.is_second_rank())
        self.assertFalse(tensor.is_fourth_rank())

        data = sp.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        tensor = SymbolicTensor(data)
        self.assertFalse(tensor.is_second_rank())
        self.assertFalse(tensor.is_fourth_rank())

        data = sp.Array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
            ]
        )
        tensor = SymbolicTensor(data)
        self.assertFalse(tensor.is_second_rank())
        self.assertTrue(tensor.is_fourth_rank())

    def test_is_square(self):
        data = sp.Array([[1, 2], [3, 4]])
        tensor = SymbolicTensor(data)
        self.assertTrue(tensor.is_square())

        data = sp.Array([[1, 2], [3, 4], [5, 6]])
        tensor = SymbolicTensor(data)
        self.assertFalse(tensor.is_square())

    def test_is_symmetric(self):
        data = sp.Array([[1, 2], [2, 3]])
        tensor = SymbolicTensor(data)
        self.assertTrue(tensor.is_symmetric())

        data = sp.Array([[1, 2], [1, 2]])
        tensor = SymbolicTensor(data)
        self.assertFalse(tensor.is_symmetric())

    def test_to_matrix(self):
        components = [[1, 2], [2, 3]]
        data = sp.Array(components)
        tensor = SymbolicTensor(data)
        matrix = tensor.to_matrix()
        self.assertIsInstance(matrix, sp.MutableDenseMatrix)
        self.assertEqual(matrix, sp.MutableDenseMatrix(components))

        components = [[1, 2, 3], [4, 5, 6]]
        data = sp.Array(components)
        tensor = SymbolicTensor(data)
        with self.assertRaises(ValueError) as context:
            tensor.to_matrix()
        self.assertEqual(
            str(context.exception),
            "The Tensor should be a second rank square.",
        )

    def test_to_3x3(self):
        data = sp.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = SymbolicTensor(data)
        new_tensor = tensor.to_3x3()
        self.assertIsInstance(new_tensor, SymbolicThreeByThreeTensor)
        self.assertEqual(new_tensor.data, data)

    def test_to_sym_3x3(self):
        data = sp.NDimArray([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        tensor = SymbolicTensor(data)
        new_tensor = tensor.to_sym_3x3()
        self.assertIsInstance(new_tensor, SymbolicSymmetricThreeByThreeTensor)
        self.assertEqual(new_tensor.data, sp.NDimArray([1, 2, 3, 4, 5, 6]))

    def test_to_6x6(self):
        data = sp.NDimArray(
            [
                [11, 21, 31, 41, 51, 61],
                [12, 22, 32, 42, 52, 62],
                [13, 23, 33, 43, 53, 63],
                [14, 24, 34, 44, 54, 64],
                [15, 25, 35, 45, 55, 65],
                [16, 26, 36, 46, 56, 66],
            ]
        )
        tensor = SymbolicTensor(data)
        new_tensor = tensor.to_6x6()
        self.assertIsInstance(new_tensor, SymbolicSixBySixTensor)
        self.assertEqual(new_tensor.data, data)

    def test_from_list(self):
        components = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tensor = SymbolicTensor.from_list(components, shape=(3, 3))
        self.assertEqual(tensor.data, sp.NDimArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        components = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        tensor = SymbolicTensor.from_list(components, shape=(3, 3))
        self.assertEqual(tensor.data, sp.NDimArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        with self.assertRaises(ValueError) as context:
            SymbolicTensor.from_list(components=None, shape=(3, 3))
        self.assertEqual(str(context.exception), "Components must be a list.")

        components = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        with self.assertRaises(ValueError) as context:
            SymbolicTensor.from_list(components, shape=None)
        self.assertEqual(str(context.exception), "Shape must be a tuple.")

    def test_create(self):
        tensor = SymbolicTensor.create(shape=(1,), name="M")
        self.assertEqual(tensor.data[0], sp.symbols("M_1"))

        tensor = SymbolicTensor.create(shape=(1, 1), name="M")
        self.assertEqual(tensor.data[0, 0], sp.symbols("M_11"))

        tensor = SymbolicTensor.create(shape=(1, 1, 1), name="M")
        self.assertEqual(tensor.data[0, 0, 0], sp.symbols("M_111"))

        tensor = SymbolicTensor.create(shape=(1, 1, 1, 1), name="M")
        self.assertEqual(tensor.data[0, 0, 0, 0], sp.symbols("M_1111"))

        tensor = SymbolicTensor.create(shape=(3, 3), name="M")
        self.assertEqual(tensor.data[0, 0], sp.symbols("M_11"))
        self.assertEqual(tensor.data[1, 2], sp.symbols("M_23"))
        self.assertEqual(tensor.data[2, 2], sp.symbols("M_33"))

    def test_subs_tensor_params(self):
        pass

    def test_susb(self):
        pass

    def test_matmul(self):
        tensor_M = SymbolicTensor.create(shape=(2, 2), name="M")
        tensor_N = SymbolicTensor.create(shape=(2, 2), name="N")
        tensor_MN = tensor_M @ tensor_N
        M_11, M_12, M_21, M_22 = sp.symbols("M_11 M_12 M_21 M_22")
        N_11, N_12, N_21, N_22 = sp.symbols("N_11 N_12 N_21 N_22")
        self.assertEqual(tensor_MN.data[0, 0], M_11 * N_11 + M_12 * N_21)
        self.assertEqual(tensor_MN.data[0, 1], M_11 * N_12 + M_12 * N_22)
        self.assertEqual(tensor_MN.data[1, 0], M_21 * N_11 + M_22 * N_21)
        self.assertEqual(tensor_MN.data[1, 1], M_21 * N_12 + M_22 * N_22)

    def test_getitem(self):
        tensor = SymbolicTensor.create(shape=(2, 2), name="M")
        self.assertEqual(tensor[0, 0], tensor.data[0, 0])
        self.assertEqual(tensor[0, 1], tensor.data[0, 1])
        self.assertEqual(tensor[1, 0], tensor.data[1, 0])
        self.assertEqual(tensor[1, 1], tensor.data[1, 1])

    def test_eigenvalues(self):
        pass

    def test_eigenvectors(self):
        pass

    def test_diagonalize(self):
        pass


class TestSymbolicThreeByThreeTensor(unittest.TestCase):
    def test_init(self):
        data = sp.NDimArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tensor = SymbolicThreeByThreeTensor(data)
        self.assertEqual(tensor.shape, tensor.data.shape)

        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # not SymPy Array
        with self.assertRaises(ValueError) as context:
            SymbolicThreeByThreeTensor(data)
        self.assertEqual(str(context.exception), "Data must be a 3x3 SymPy Array.")

        data = sp.NDimArray([[1, 2], [3, 4]])  # not 3x3
        with self.assertRaises(ValueError) as context:
            SymbolicThreeByThreeTensor(data)
        self.assertEqual(str(context.exception), "Data must be a 3x3 SymPy Array.")

        data = [[1, 2], [3, 4]]  # not 3x3 SymPy Array
        with self.assertRaises(ValueError) as context:
            SymbolicThreeByThreeTensor(data)
        self.assertEqual(str(context.exception), "Data must be a 3x3 SymPy Array.")

    def test_form_list(self):
        components = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tensor = SymbolicThreeByThreeTensor.from_list(components)
        self.assertEqual(tensor.data.shape, (3, 3))
        self.assertEqual(tensor.data, sp.NDimArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        components = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        tensor = SymbolicThreeByThreeTensor.from_list(components)
        self.assertEqual(tensor.data.shape, (3, 3))
        self.assertEqual(tensor.data, sp.NDimArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        with self.assertRaises(ValueError) as context:
            SymbolicThreeByThreeTensor.from_list(components=None)
        self.assertEqual(str(context.exception), "Components must be a list.")

    def test_create(self):
        tensor = SymbolicThreeByThreeTensor.create(name="M")
        self.assertEqual(tensor.data.shape, (3, 3))
        self.assertEqual(tensor.data[0, 0], sp.symbols("M_11"))
        self.assertEqual(tensor.data[1, 2], sp.symbols("M_23"))
        self.assertEqual(tensor.data[2, 2], sp.symbols("M_33"))

    def test_to_symmetric(self):
        data = sp.NDimArray(
            [
                [1, 2, 3],
                [2, 4, 5],
                [3, 5, 6],
            ]
        )
        tensor = SymbolicThreeByThreeTensor(data)
        sym_tensor = tensor.to_symmetric()
        self.assertEqual(type(sym_tensor), SymbolicSymmetricThreeByThreeTensor)
        self.assertEqual(sym_tensor.data, sp.NDimArray([1, 2, 3, 4, 5, 6]))

        data = sp.NDimArray(
            [
                [1, 6, 5],
                [6, 2, 4],
                [5, 4, 3],
            ]
        )
        tensor = SymbolicThreeByThreeTensor(data)
        sym_tensor = tensor.to_symmetric(notation="voigt")
        self.assertEqual(type(sym_tensor), SymbolicSymmetricThreeByThreeTensor)
        self.assertEqual(sym_tensor.data, sp.NDimArray([1, 2, 3, 4, 5, 6]))

        data = sp.NDimArray(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        tensor = SymbolicThreeByThreeTensor(data)
        with self.assertRaises(ValueError) as context:
            tensor.to_symmetric()
        self.assertEqual(str(context.exception), "The tensor is not symmetric.")

        data = sp.NDimArray(
            [
                [1, 2, 3],
                [2, 4, 5],
                [3, 5, 6],
            ]
        )
        tensor = SymbolicThreeByThreeTensor(data)
        with self.assertRaises(NotImplementedError) as context:
            tensor.to_symmetric(notation="NOT_IMPLEMENTED_NOTATION")
        self.assertEqual(
            str(context.exception), "Notation NOT_IMPLEMENTED_NOTATION not implemented"
        )


class TestSymbolicSixBySixTensor(unittest.TestCase):
    def test_init(self):
        data = sp.NDimArray(
            [
                [11, 21, 31, 41, 51, 61],
                [12, 22, 32, 42, 52, 62],
                [13, 23, 33, 43, 53, 63],
                [14, 24, 34, 44, 54, 64],
                [15, 25, 35, 45, 55, 65],
                [16, 26, 36, 46, 56, 66],
            ]
        )
        tensor = SymbolicSixBySixTensor(data)
        self.assertEqual(tensor.shape, tensor.data.shape)

        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # not 6x6 SymPy Array
        with self.assertRaises(ValueError) as context:
            SymbolicSixBySixTensor(data)
        self.assertEqual(str(context.exception), "Data must be a 6x6 SymPy Array.")

    def test_form_list(self):
        components = [
            [11, 21, 31, 41, 51, 61],
            [12, 22, 32, 42, 52, 62],
            [13, 23, 33, 43, 53, 63],
            [14, 24, 34, 44, 54, 64],
            [15, 25, 35, 45, 55, 65],
            [16, 26, 36, 46, 56, 66],
        ]
        tensor = SymbolicSixBySixTensor.from_list(components)
        self.assertEqual(tensor.data.shape, (6, 6))
        self.assertEqual(
            tensor.data,
            sp.NDimArray(
                [
                    [11, 21, 31, 41, 51, 61],
                    [12, 22, 32, 42, 52, 62],
                    [13, 23, 33, 43, 53, 63],
                    [14, 24, 34, 44, 54, 64],
                    [15, 25, 35, 45, 55, 65],
                    [16, 26, 36, 46, 56, 66],
                ]
            ),
        )

        components = [
            11,
            21,
            31,
            41,
            51,
            61,
            12,
            22,
            32,
            42,
            52,
            62,
            13,
            23,
            33,
            43,
            53,
            63,
            14,
            24,
            34,
            44,
            54,
            64,
            15,
            25,
            35,
            45,
            55,
            65,
            16,
            26,
            36,
            46,
            56,
            66,
        ]
        tensor = SymbolicSixBySixTensor.from_list(components)
        self.assertEqual(tensor.data.shape, (6, 6))
        self.assertEqual(
            tensor.data,
            sp.NDimArray(
                [
                    [11, 21, 31, 41, 51, 61],
                    [12, 22, 32, 42, 52, 62],
                    [13, 23, 33, 43, 53, 63],
                    [14, 24, 34, 44, 54, 64],
                    [15, 25, 35, 45, 55, 65],
                    [16, 26, 36, 46, 56, 66],
                ]
            ),
        )

    def test_create(self):
        tensor = SymbolicSixBySixTensor.create(name="M")
        self.assertEqual(tensor.data.shape, (6, 6))
        self.assertEqual(tensor.data[0, 0], sp.symbols("M_11"))
        self.assertEqual(tensor.data[3, 4], sp.symbols("M_45"))
        self.assertEqual(tensor.data[5, 5], sp.symbols("M_66"))


class TestSymbolicSymmetricThreeByThreeTensor(unittest.TestCase):
    def test_init(self):
        data = sp.NDimArray([1, 2, 3, 4, 5, 6])
        tensor = SymbolicSymmetricThreeByThreeTensor(data)
        self.assertEqual(tensor.data.shape, (6,))
        self.assertEqual(tensor.data, data)
        self.assertEqual(tensor.name, None)
        self.assertEqual(tensor.notation, "standard")
        self.assertEqual(tensor.tensor_params, {})

        data = sp.NDimArray([1, 2, 3, 4, 5, 6])
        name = sp.symbols("M")
        notation = "voigt"
        n, m = sp.symbols("n m")
        tensor_params = {n: {1, 2, 3}, m: {4, 5, 6}}
        tensor = SymbolicSymmetricThreeByThreeTensor(
            data,
            name,
            notation,
            tensor_params,
        )
        self.assertEqual(tensor.data.shape, (6,))
        self.assertEqual(tensor.data, data)
        self.assertEqual(tensor.name, name)
        self.assertEqual(tensor.notation, notation)
        self.assertEqual(tensor.tensor_params, tensor_params)

        data = sp.NDimArray([1, 2, 3, 4])
        with self.assertRaises(ValueError) as context:
            SymbolicSymmetricThreeByThreeTensor(data)
        self.assertEqual(
            str(context.exception),
            "Data must be a 6x1 SymPy Array.",
        )

        data = sp.NDimArray([1, 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError) as context:
            SymbolicSymmetricThreeByThreeTensor(data)
        self.assertEqual(
            str(context.exception),
            "Data must be a 6x1 SymPy Array.",
        )

        data = sp.NDimArray([1, 2, 3, 4, 5, 6])
        notation = "NOT_IMPLEMENTED_NOTATION"
        with self.assertRaises(NotImplementedError) as context:
            SymbolicSymmetricThreeByThreeTensor(data, notation=notation)
        self.assertEqual(
            str(context.exception),
            "Notation NOT_IMPLEMENTED_NOTATION not implemented.",
        )

    def test_from_list(self):
        components = [1, 2, 3, 4, 5, 6]
        tensor = SymbolicSymmetricThreeByThreeTensor.from_list(components)
        self.assertIsInstance(tensor, SymbolicSymmetricThreeByThreeTensor)
        self.assertEqual(tensor.data.shape, (6,))
        self.assertEqual(tensor.data, sp.NDimArray([1, 2, 3, 4, 5, 6]))
        self.assertEqual(tensor.notation, "standard")

        components = [1, 2, 3, 2, 4, 5, 3, 5, 6]
        tensor = SymbolicSymmetricThreeByThreeTensor.from_list(
            components,
            notation="standard",
        )
        self.assertIsInstance(tensor, SymbolicSymmetricThreeByThreeTensor)
        self.assertEqual(tensor.data.shape, (6,))
        self.assertEqual(tensor.data, sp.NDimArray([1, 2, 3, 4, 5, 6]))
        self.assertEqual(tensor.notation, "standard")

        components = [1, 6, 5, 6, 2, 4, 5, 4, 3]
        tensor = SymbolicSymmetricThreeByThreeTensor.from_list(
            components,
            notation="voigt",
        )
        self.assertIsInstance(tensor, SymbolicSymmetricThreeByThreeTensor)
        self.assertEqual(tensor.data.shape, (6,))
        self.assertEqual(tensor.data, sp.NDimArray([1, 2, 3, 4, 5, 6]))
        self.assertEqual(tensor.notation, "voigt")

        components = [1, 2, 1, 2, 4, 5, 3, 5, 6]
        with self.assertRaises(ValueError) as context:
            SymbolicSymmetricThreeByThreeTensor.from_list(components)
        self.assertEqual(
            str(context.exception),
            "The tensor is not symmetric.",
        )

    def test_create(self):
        name = sp.symbols("M")
        M_1, M_2, M_3, M_4, M_5, M_6 = sp.symbols("M_1 M_2 M_3 M_4 M_5 M_6")
        M_11, M_22, M_33, M_23, M_13, M_12 = sp.symbols("M_11 M_22 M_33 M_23 M_13 M_12")

        tensor = SymbolicSymmetricThreeByThreeTensor.create(name)
        self.assertEqual(tensor.data, sp.NDimArray([M_1, M_2, M_3, M_4, M_5, M_6]))

        tensor = SymbolicSymmetricThreeByThreeTensor.create(name, notation="voigt")
        self.assertEqual(
            tensor.data, sp.NDimArray([M_11, M_22, M_33, M_23, M_13, M_12])
        )

    def test_to_general(self):
        name = sp.symbols("M")
        M_1, M_2, M_3, M_4, M_5, M_6 = sp.symbols("M_1 M_2 M_3 M_4 M_5 M_6")
        M_11, M_22, M_33, M_23, M_13, M_12 = sp.symbols("M_11 M_22 M_33 M_23 M_13 M_12")

        tensor = SymbolicSymmetricThreeByThreeTensor.create(name).to_general()
        self.assertIsInstance(tensor, SymbolicThreeByThreeTensor)
        self.assertEqual(
            tensor.data,
            sp.NDimArray(
                [
                    [M_1, M_2, M_3],
                    [M_2, M_4, M_5],
                    [M_3, M_5, M_6],
                ]
            ),
        )

        tensor = SymbolicSymmetricThreeByThreeTensor.create(
            name,
            notation="voigt",
        ).to_general()
        self.assertIsInstance(tensor, SymbolicThreeByThreeTensor)
        self.assertEqual(
            tensor.data,
            sp.NDimArray(
                [
                    [M_11, M_12, M_13],
                    [M_12, M_22, M_23],
                    [M_13, M_23, M_33],
                ]
            ),
        )

    def test_getitem(self):
        M_1, M_2, M_3, M_4, M_5, M_6 = sp.symbols("M_1 M_2 M_3 M_4 M_5 M_6")

        tensor = SymbolicSymmetricThreeByThreeTensor.from_list(
            [M_1, M_2, M_3, M_4, M_5, M_6]
        )
        self.assertEqual(tensor[0], M_1)
        self.assertEqual(tensor[1], M_2)
        self.assertEqual(tensor[2], M_3)
        self.assertEqual(tensor[3], M_4)
        self.assertEqual(tensor[4], M_5)
        self.assertEqual(tensor[5], M_6)

        self.assertEqual(tensor[0, 0], M_1)
        self.assertEqual(tensor[1, 1], M_4)
        self.assertEqual(tensor[2, 2], M_6)

        self.assertEqual(tensor[0, 0], M_1)
        self.assertEqual(tensor[0, 1], M_2)
        self.assertEqual(tensor[0, 2], M_3)

        tensor = SymbolicSymmetricThreeByThreeTensor.from_list(
            [M_1, M_2, M_3, M_4, M_5, M_6],
            notation="voigt",
        )
        self.assertEqual(tensor[0], M_1)
        self.assertEqual(tensor[1], M_2)
        self.assertEqual(tensor[2], M_3)
        self.assertEqual(tensor[3], M_4)
        self.assertEqual(tensor[4], M_5)
        self.assertEqual(tensor[5], M_6)

        self.assertEqual(tensor[0, 0], M_1)
        self.assertEqual(tensor[1, 1], M_2)
        self.assertEqual(tensor[2, 2], M_3)

        self.assertEqual(tensor[0, 0], M_1)
        self.assertEqual(tensor[0, 1], M_6)
        self.assertEqual(tensor[0, 2], M_5)

    def test_eigenvalues(self):
        pass

    def test_eigenvectors(self):
        pass

    def test_diagonalize(self):
        pass


class TestSymbolicStressTensor(unittest.TestCase):
    def test_init(self):
        s_1, s_2, s_3, s_4, s_5, s_6 = sp.symbols(
            "\\sigma_1 \\sigma_2 \\sigma_3 \\sigma_4 \\sigma_5 \\sigma_6"
        )
        data = sp.NDimArray([s_1, s_2, s_3, s_4, s_5, s_6])
        stress_tensor = SymbolicStressTensor(data)
        self.assertEqual(stress_tensor.data, data)
        self.assertEqual(stress_tensor.name, None)
        self.assertEqual(stress_tensor.notation, "standard")
        self.assertEqual(stress_tensor.tensor_params, {})

    def test_create(self):
        s_1, s_2, s_3, s_4, s_5, s_6 = sp.symbols(
            "\\sigma_1 \\sigma_2 \\sigma_3 \\sigma_4 \\sigma_5 \\sigma_6"
        )
        data = sp.NDimArray([s_1, s_2, s_3, s_4, s_5, s_6])

        stress_tensor = SymbolicStressTensor.create()
        self.assertIsInstance(stress_tensor, SymbolicStressTensor)
        self.assertEqual(stress_tensor.data, data)

    def test_to_general(self):
        pass

    def test_normal_components(self):
        s_1, s_2, s_3, s_4, s_5, s_6 = sp.symbols(
            "\\sigma_1 \\sigma_2 \\sigma_3 \\sigma_4 \\sigma_5 \\sigma_6"
        )
        data = sp.NDimArray([s_1, s_2, s_3, s_4, s_5, s_6])
        stress_tensor = SymbolicStressTensor(data)
        self.assertEqual(
            stress_tensor.normal_components(),
            sp.NDimArray([s_1, s_2, s_3]),
        )

    def test_shear_components(self):
        s_1, s_2, s_3, s_4, s_5, s_6 = sp.symbols(
            "\\sigma_1 \\sigma_2 \\sigma_3 \\sigma_4 \\sigma_5 \\sigma_6"
        )
        data = sp.NDimArray([s_1, s_2, s_3, s_4, s_5, s_6])
        stress_tensor = SymbolicStressTensor(data)
        self.assertEqual(
            stress_tensor.shear_components(),
            sp.NDimArray([s_4, s_5, s_6]),
        )

    def test_principal_components(self):
        pass

    def test_pressure(self):
        pass

    def test_tresca(self):
        pass

    def test_von_mises(self):
        pass


class TestSymbolicStrainTensor(unittest.TestCase):
    def test_init(self):
        s_1, s_2, s_3, s_4, s_5, s_6 = sp.symbols(
            "\\epsilon_1 \\epsilon_2 \\epsilon_3 \\epsilon_4 \\epsilon_5 \\epsilon_6"
        )
        data = sp.NDimArray([s_1, s_2, s_3, s_4, s_5, s_6])
        strain_tensor = SymbolicStrainTensor(data)
        self.assertEqual(strain_tensor.data, data)
        self.assertEqual(strain_tensor.name, None)
        self.assertEqual(strain_tensor.notation, "standard")
        self.assertEqual(strain_tensor.tensor_params, {})

    def test_create(self):
        s_1, s_2, s_3, s_4, s_5, s_6 = sp.symbols(
            "\\epsilon_1 \\epsilon_2 \\epsilon_3 \\epsilon_4 \\epsilon_5 \\epsilon_6"
        )
        data = sp.NDimArray([s_1, s_2, s_3, s_4, s_5, s_6])

        strain_tensor = SymbolicStrainTensor.create()
        self.assertIsInstance(strain_tensor, SymbolicStrainTensor)
        self.assertEqual(strain_tensor.data, data)

    def test_to_general(self):
        pass

    def test_normal_components(self):
        pass

    def test_shear_components(self):
        pass

    def test_principal_components(self):
        pass

    def test_volumetric_strain(self):
        pass


class TestSymbolicDisplacement(unittest.TestCase):
    def test_init(self):
        coord_system = SymbolicCartesianCoordSystem()
        data = sp.NDimArray([1, 2, 3])
        displacement_field = SymbolicDisplacement(coord_system=coord_system, data=data)
        self.assertEqual(displacement_field.data, data)

    def test_strain_tensor(self):
        coord_system = SymbolicCartesianCoordSystem()
        x1, x2, x3 = coord_system.basis
        data = sp.NDimArray([x1, x2, x3])
        displacement_field = SymbolicDisplacement(coord_system=coord_system, data=data)
        strain_tensor = displacement_field.strain_tensor()
        self.assertEqual(strain_tensor.data, sp.NDimArray([1, 1, 1, 0, 0, 0]))

        data = sp.NDimArray([x2, x3, x1])
        displacement_field = SymbolicDisplacement(coord_system=coord_system, data=data)
        strain_tensor = displacement_field.strain_tensor()
        self.assertEqual(strain_tensor.data, sp.NDimArray([0, 0, 0, 1, 1, 1]))

        data = sp.NDimArray([x3, x2, x1])
        displacement_field = SymbolicDisplacement(coord_system=coord_system, data=data)
        strain_tensor = displacement_field.strain_tensor()
        self.assertEqual(strain_tensor.data, sp.NDimArray([0, 1, 0, 0, 0, 2]))


if __name__ == "__main__":
    unittest.main()
