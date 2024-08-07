import unittest
import numpy as np
import sympy as sp

from mechpy.core.symbolic.material import SymbolicIsotropicMaterial
from mechpy.core.symbolic.stress import SymbolicStressTensor
from mechpy.core.symbolic.strain import SymbolicStrainTensor
from mechpy.core.symbolic.elasticity import SymbolicElasticity


class TestSymbolicElasticity(unittest.TestCase):
    def test_hookes_law(self):
        material_props = {"E": sp.symbols("E"), "nu": sp.symbols("nu")}
        material = SymbolicIsotropicMaterial(**material_props)
        stress_tensor = SymbolicStressTensor.create(notation="voigt")
        components_values = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }
        stress_tensor.subs_tensor_components(components_values)
        elasticity = SymbolicElasticity(material=material, stress_tensor=stress_tensor)
        strain_tensor = elasticity.compute_strain()
        sigma_11 = stress_tensor[0, 0]
        E = material.E
        nu = material.nu
        self.assertEqual(strain_tensor[0], sigma_11 / E)
        self.assertEqual(strain_tensor[1], -((sigma_11 * nu) / E))
        self.assertEqual(strain_tensor[2], -((sigma_11 * nu) / E))
        self.assertEqual(strain_tensor[3], 0)
        self.assertEqual(strain_tensor[4], 0)
        self.assertEqual(strain_tensor[5], 0)

    def test_hookes_law_inverse(self):
        material_props = {"E": sp.symbols("E"), "nu": sp.symbols("nu")}
        material = SymbolicIsotropicMaterial(**material_props)
        strain_tensor = SymbolicStrainTensor.create(notation="voigt")
        components_values = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }
        strain_tensor.subs_tensor_components(components_values)
        elasticity = SymbolicElasticity(material=material, strain_tensor=strain_tensor)
        stress_tensor = elasticity.compute_stress()
        epsilon_11 = strain_tensor[0, 0]
        E = material.E
        nu = material.nu
        self.assertEqual(
            stress_tensor[0],
            ((E * epsilon_11 * (nu - 1)) / ((nu + 1) * (2 * nu - 1))),
        )
        self.assertEqual(
            stress_tensor[1],
            -((E * epsilon_11 * nu) / ((nu + 1) * (2 * nu - 1))),
        )
        self.assertEqual(
            stress_tensor[2],
            -((E * epsilon_11 * nu) / ((nu + 1) * (2 * nu - 1))),
        )
        self.assertEqual(stress_tensor[3], 0)
        self.assertEqual(stress_tensor[4], 0)
        self.assertEqual(stress_tensor[5], 0)


if __name__ == "__main__":
    unittest.main()
