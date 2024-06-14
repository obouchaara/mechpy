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

if __name__ == "__main__":
    unittest.main()
