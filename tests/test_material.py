import unittest
import numpy as np
import sympy as sp

from mechpy.core.symbolic.material import (
    SymbolicComplianceTensor,
    SymbolicStiffnessTensor,
    SymbolicMaterial,
    SymbolicElasticMaterial,
    SymbolicIsotropicMaterial,
    SymbolicTransverseIsotropicMaterial,
    SymbolicOrthotropicMaterial,
)


class TestSymbolicMaterial(unittest.TestCase):
    def test_init(self):
        material_props = {"E": 200 * 10**12, "nu": 0.3}
        material = SymbolicMaterial(**material_props)
        self.assertTrue("E" in material.__dict__.keys())
        self.assertEqual(material.E, material_props["E"])
        self.assertEqual(material.nu, material_props["nu"])


class TestSymbolicElasticMaterial(unittest.TestCase):
    def test_compliance_tensor(self):
        pass


class TestSymbolicIsotropicMaterial(unittest.TestCase):
    def test_init(self):
        E, nu = sp.symbols("E nu")
        material_props = {"E": E, "nu": nu}
        material = SymbolicIsotropicMaterial(**material_props)
        self.assertEqual(
            material.__getattribute__("E"),
            material_props["E"],
        )
        self.assertEqual(
            material.__getattribute__("nu"),
            material_props["nu"],
        )

        lamda, mu = sp.symbols("lamda mu")
        material_props = {"lamda": lamda, "mu": mu}
        material = SymbolicIsotropicMaterial(**material_props)
        self.assertEqual(
            material.__getattribute__("lamda"),
            material_props["lamda"],
        )
        self.assertEqual(
            material.__getattribute__("mu"),
            material_props["mu"],
        )

        with self.assertRaises(ValueError) as context:
            SymbolicIsotropicMaterial()
        self.assertEqual(
            str(context.exception),
            "Material properties must include either both 'E' and 'nu' or both 'lamda' and 'mu'",
        )

        E, mu = sp.symbols("E mu")
        material_props = {"E": E, "mu": mu}
        with self.assertRaises(ValueError) as context:
            SymbolicIsotropicMaterial(**material_props)
        self.assertEqual(
            str(context.exception),
            "Material properties must include either both 'E' and 'nu' or both 'lamda' and 'mu'",
        )

    def test_lame_params(self):
        E, nu = sp.symbols("E nu")
        lamda, mu = SymbolicIsotropicMaterial.lame_params(E, nu)
        expected_lamda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        expected_mu = E / (2 * (1 + nu))
        self.assertEqual(lamda, expected_lamda)
        self.assertEqual(mu, expected_mu)

    def test_youngs_params(self):
        lamda, mu = sp.symbols("lamda mu")
        E, nu = SymbolicIsotropicMaterial.youngs_params(lamda, mu)
        self.assertEqual(E, mu * (3 * lamda + 2 * mu) / (lamda + mu))
        self.assertEqual(nu, lamda / (2 * (lamda + mu)))

    def test_stiffness_tensor(self):
        lamda, mu = sp.symbols("lamda mu")
        material = SymbolicIsotropicMaterial(lamda=lamda, mu=mu)
        tensor = material.stiffness_tensor()
        self.assertIsInstance(tensor, SymbolicStiffnessTensor)
        C_11 = lamda + 2 * mu
        C_12 = lamda
        C_44 = mu
        expected_data = sp.sympify(
            sp.NDimArray(
                [
                    [C_11, C_12, C_12, 0, 0, 0],
                    [C_12, C_11, C_12, 0, 0, 0],
                    [C_12, C_12, C_11, 0, 0, 0],
                    [0, 0, 0, C_44, 0, 0],
                    [0, 0, 0, 0, C_44, 0],
                    [0, 0, 0, 0, 0, C_44],
                ]
            )
        )
        self.assertEqual(tensor.data, expected_data)

        E, nu = sp.symbols("E nu")
        material = SymbolicIsotropicMaterial(E=E, nu=nu)
        tensor = material.stiffness_tensor()
        self.assertIsInstance(tensor, SymbolicStiffnessTensor)
        lamda, mu = SymbolicIsotropicMaterial.lame_params(E, nu)
        C_11 = lamda + 2 * mu
        C_12 = lamda
        C_44 = mu
        expected_data = sp.factor(
            sp.NDimArray(
                [
                    [C_11, C_12, C_12, 0, 0, 0],
                    [C_12, C_11, C_12, 0, 0, 0],
                    [C_12, C_12, C_11, 0, 0, 0],
                    [0, 0, 0, C_44, 0, 0],
                    [0, 0, 0, 0, C_44, 0],
                    [0, 0, 0, 0, 0, C_44],
                ]
            )
        )
        self.assertEqual(tensor.data, expected_data)

    def test_compliance_tensor(self):
        E, nu = sp.symbols("E nu")
        material = SymbolicIsotropicMaterial(E=E, nu=nu)
        tensor = material.compliance_tensor()
        self.assertIsInstance(tensor, SymbolicComplianceTensor)
        S_11 = 1 / E
        S_12 = -(nu / E)
        S_44 = (2 * (1 + nu)) / E
        expected_data = sp.factor(
            sp.NDimArray(
                [
                    [S_11, S_12, S_12, 0, 0, 0],
                    [S_12, S_11, S_12, 0, 0, 0],
                    [S_12, S_12, S_11, 0, 0, 0],
                    [0, 0, 0, S_44, 0, 0],
                    [0, 0, 0, 0, S_44, 0],
                    [0, 0, 0, 0, 0, S_44],
                ]
            )
        )
        self.assertEqual(tensor.data, expected_data)

        lamda, mu = sp.symbols("lamda mu")
        material = SymbolicIsotropicMaterial(lamda=lamda, mu=mu)
        tensor = material.compliance_tensor()
        E, nu = SymbolicIsotropicMaterial.youngs_params(lamda, mu)
        self.assertIsInstance(tensor, SymbolicComplianceTensor)
        S_11 = 1 / E
        S_12 = -(nu / E)
        S_44 = (2 * (1 + nu)) / E
        expected_data = sp.factor(
            sp.NDimArray(
                [
                    [S_11, S_12, S_12, 0, 0, 0],
                    [S_12, S_11, S_12, 0, 0, 0],
                    [S_12, S_12, S_11, 0, 0, 0],
                    [0, 0, 0, S_44, 0, 0],
                    [0, 0, 0, 0, S_44, 0],
                    [0, 0, 0, 0, 0, S_44],
                ]
            )
        )
        self.assertEqual(tensor.data, expected_data)


class TestSymbolicTransverseIsotropicMaterial(unittest.TestCase):
    def test_init(self):
        cls = SymbolicTransverseIsotropicMaterial
        material_props = {_: sp.symbols(_) for _ in cls.props_keys}
        material = cls(**material_props)
        for key in material_props.keys():
            self.assertEqual(
                material.__getattribute__(key),
                material_props[key],
            )

    def test_stiffness_tensor(self):
        cls = SymbolicTransverseIsotropicMaterial
        material_props = {_: sp.symbols(_) for _ in cls.props_keys}
        material = cls(**material_props)
        tensor = material.stiffness_tensor()
        self.assertIsInstance(tensor, SymbolicStiffnessTensor)

        E_L = material_props["E_L"]
        E_T = material_props["E_T"]
        nu = material_props["nu"]
        G_L = material_props["G_L"]
        G_T = material_props["G_T"]

        C_11 = E_L / (1 - nu**2)
        C_12 = E_L * nu / (1 - nu)
        C_33 = E_T
        C_44 = G_L
        C_66 = G_T

        expected_data = sp.simplify(
            sp.NDimArray(
                [
                    [C_11, C_12, C_12, 0, 0, 0],
                    [C_12, C_11, C_12, 0, 0, 0],
                    [C_12, C_12, C_33, 0, 0, 0],
                    [0, 0, 0, C_44, 0, 0],
                    [0, 0, 0, 0, C_44, 0],
                    [0, 0, 0, 0, 0, C_66],
                ]
            )
        )
        self.assertEqual(tensor.data, expected_data)


class TestSymbolicOrthotropicMaterial(unittest.TestCase):
    def test_init(self):
        cls = SymbolicOrthotropicMaterial
        material_props = {_: sp.symbols(_) for _ in cls.props_keys}
        material = cls(**material_props)
        for key in material_props.keys():
            self.assertEqual(
                material.__getattribute__(key),
                material_props[key],
            )

    def test_stiffness_tensor(self):
        cls = SymbolicOrthotropicMaterial
        material_props = {_: sp.symbols(_) for _ in cls.props_keys}
        material = cls(**material_props)
        tensor = material.stiffness_tensor()
        self.assertIsInstance(tensor, SymbolicStiffnessTensor)

        C11 = material_props["E1"]
        C22 = material_props["E2"]
        C33 = material_props["E3"]
        C44 = material_props["G23"]
        C55 = material_props["G31"]
        C66 = material_props["G12"]
        C12 = material_props["nu12"] * C22
        C13 = material_props["nu31"] * C11
        C23 = material_props["nu23"] * C33

        expected_data = sp.sympify(
            sp.NDimArray(
                [
                    [C11, C12, C13, 0, 0, 0],
                    [C12, C22, C23, 0, 0, 0],
                    [C13, C23, C33, 0, 0, 0],
                    [0, 0, 0, C44, 0, 0],
                    [0, 0, 0, 0, C55, 0],
                    [0, 0, 0, 0, 0, C66],
                ]
            )
        )
        self.assertEqual(tensor.data, expected_data)


if __name__ == "__main__":
    unittest.main()
