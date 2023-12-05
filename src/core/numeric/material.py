import numpy as np

from .tensor import SixBySixTensor


class Material:
    def __init__(self, mechanical_props=None, thermic_props=None):
        self.mechanical_props = mechanical_props or {}
        self.thermic_props = thermic_props or {}

    def __repr__(self):
        return f"Material(mechanical_props={self.mechanical_props}, thermic_props={self.thermic_props})"


class ElasticMaterial(Material):
    def __init__(self, mechanical_props):
        super().__init__(mechanical_props)

    def __repr__(self):
        return f"ElasticMaterial()"

    def compliance_tensor(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def stiffness_tensor(self):
        raise NotImplementedError("Subclasses must implement this method.")


class ComplianceTensor(SixBySixTensor):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"ComplianceTensor(\n{self.data}\n)"


class StiffnessTensor(SixBySixTensor):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"StiffnessTensor(\n{self.data}\n)"


class IsotropicMaterial(ElasticMaterial):
    def __init__(
        self, youngs_modulus=None, poisson_ratio=None, lames_lambda=None, lames_mu=None
    ):
        if youngs_modulus and poisson_ratio:
            self.youngs_modulus = youngs_modulus
            self.poisson_ratio = poisson_ratio

        elif lames_lambda and lames_mu:
            lamda = lames_lambda
            mu = lames_mu
            self.youngs_modulus = mu * (3 * lamda + 2 * mu) / (lamda + mu)
            self.poisson_ratio = lamda / (2 * (lamda + mu))
        else:
            raise ValueError(
                "Either Young's modulus and Poisson's ratio or Lame's lambda and Lame's mu must be provided."
            )
        mechanical_props = {
            "youngs_modulus": self.youngs_modulus,
            "poisson_ratio": self.poisson_ratio,
        }
        super().__init__(mechanical_props=mechanical_props)

    def __repr__(self):
        return f"IsotropicMaterial({self.youngs_modulus}, {self.poisson_ratio})"

    def get_lame_params(self):
        E = self.youngs_modulus
        nu = self.poisson_ratio
        lamda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return (lamda, mu)

    def compliance_tensor(self) -> ComplianceTensor:
        E = self.youngs_modulus
        nu = self.poisson_ratio
        S_11 = 1 / E
        S_12 = -(nu / E)
        S_44 = (2 * (1 + nu)) / E

        S = np.array(
            [
                [S_11, S_12, S_12, 0, 0, 0],
                [S_12, S_11, S_12, 0, 0, 0],
                [S_12, S_12, S_11, 0, 0, 0],
                [0, 0, 0, S_44, 0, 0],
                [0, 0, 0, 0, S_44, 0],
                [0, 0, 0, 0, 0, S_44],
            ]
        )

        return ComplianceTensor(S)

    def stiffness_tensor(self) -> StiffnessTensor:
        lamda, mu = self.get_lame_params()
        C_11 = lamda + 2 * mu
        C_12 = lamda
        C_44 = mu

        C = np.array(
            [
                [C_11, C_12, C_12, 0, 0, 0],
                [C_12, C_11, C_12, 0, 0, 0],
                [C_12, C_12, C_11, 0, 0, 0],
                [0, 0, 0, C_44, 0, 0],
                [0, 0, 0, 0, C_44, 0],
                [0, 0, 0, 0, 0, C_44],
            ]
        )

        return StiffnessTensor(C)


class TransverseIsotropicMaterial(ElasticMaterial):
    def __init__(
        self,
        youngs_modulus_parallel,
        youngs_modulus_transverse,
        poisson_ratio,
        shear_modulus,
    ):
        self.youngs_modulus_parallel = youngs_modulus_parallel
        self.youngs_modulus_transverse = youngs_modulus_transverse
        self.poisson_ratio = poisson_ratio
        self.shear_modulus = shear_modulus

        mechanical_props = {
            "youngs_modulus_parallel": self.youngs_modulus_parallel,
            "youngs_modulus_transverse": self.youngs_modulus_transverse,
            "poisson_ratio": self.poisson_ratio,
            "shear_modulus": self.shear_modulus,
        }

        super().__init__(mechanical_props=mechanical_props)

    def __repr__(self):
        return (
            f"TransverseIsotropicMaterial({self.youngs_modulus_parallel}, "
            f"{self.youngs_modulus_transverse}, {self.poisson_ratio}, "
            f"{self.shear_modulus})"
        )

    def compliance_tensor(self) -> ComplianceTensor:
        pass

    def stiffness_tensor(self) -> StiffnessTensor:
        E_L = self.youngs_modulus_parallel
        E_T = self.youngs_modulus_transverse
        nu = self.poisson_ratio
        G = self.shear_modulus

        C_11 = E_L / (1 - nu**2)
        C_12 = E_L * nu / (1 - nu)
        C_33 = E_T
        C_44 = G
        C_66 = E_T / (2 * (1 + nu))

        C = np.array(
            [
                [C_11, C_12, C_12, 0, 0, 0],
                [C_12, C_11, C_12, 0, 0, 0],
                [C_12, C_12, C_33, 0, 0, 0],
                [0, 0, 0, C_44, 0, 0],
                [0, 0, 0, 0, C_44, 0],
                [0, 0, 0, 0, 0, C_66],
            ]
        )

        return StiffnessTensor(C)


class AnisotropicMaterial(ElasticMaterial):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"AnisotropicMaterial()"

    def compliance_tensor(self) -> ComplianceTensor:
        pass

    def stiffness_tensor(self) -> StiffnessTensor:
        pass
