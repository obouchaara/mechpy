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


class AnisotropicMaterial(ElasticMaterial):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"AnisotropicMaterial()"

    def compliance_tensor(self) -> ComplianceTensor:
        pass

    def stiffness_tensor(self) -> StiffnessTensor:
        pass


class IsotropicMaterial(ElasticMaterial):
    def __init__(
        self, youngs_modulus=None, poisson_ratio=None, lames_lambda=None, lames_mu=None
    ):
        if youngs_modulus and poisson_ratio:
            self.youngs_modulus = youngs_modulus
            self.poisson_ratio = poisson_ratio

        elif lames_lambda and lames_mu:
            self.youngs_modulus = 2 * lames_mu * (1 + lames_lambda)
            self.poisson_ratio = (lames_lambda) / (2 * (1 + lames_lambda))

        else:
            raise ValueError(
                "Either Young's modulus and Poisson's ratio or Lame's lambda and Lame's mu must be provided."
            )
        super().__init__(
            mechanical_props={
                "youngs_modulus": self.youngs_modulus,
                "poisson_ratio": self.poisson_ratio,
            }
        )

    def __repr__(self):
        return f"IsotropicMaterial({self.youngs_modulus}, {self.poisson_ratio})"

    def get_lame_params(self):
        E = self.youngs_modulus
        v = self.poisson_ratio
        lames_lambda = E * v / (1 - 2 * v)
        lames_mu = E / 2 / (1 + v)
        return (lames_lambda, lames_mu)

    def compliance_tensor(self):
        lames_lambda, lames_mu = self.get_lame_params()
        C_11 = lames_lambda + 2 * lames_mu
        C_12 = lames_lambda
        C_44 = lames_mu

        return ComplianceTensor(
            np.array(
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

    def stiffness_tensor(self):
        E = self.youngs_modulus
        v = self.poisson_ratio
        S_11 = 1 / E
        S_12 = -(v / E)
        S_44 = (2 * (1 + v)) / E

        return StiffnessTensor(
            np.array(
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


class TransverseIsotropicMaterial(ElasticMaterial):
    def __init__(
        self,
        youngs_modulus_parallel,
        youngs_modulus_transverse,
        poisson_ratio_transverse,
        shear_modulus,
    ):
        self.youngs_modulus_parallel = youngs_modulus_parallel
        self.youngs_modulus_transverse = youngs_modulus_transverse
        self.poisson_ratio_transverse = poisson_ratio_transverse
        self.shear_modulus = shear_modulus

        mechanical_props = {
            youngs_modulus_parallel: youngs_modulus_parallel,
            youngs_modulus_transverse: youngs_modulus_transverse,
            poisson_ratio_transverse: poisson_ratio_transverse,
            shear_modulus: shear_modulus,
        }

        super().__init__(mechanical_props=mechanical_props)

    def __repr__(self):
        return (
            f"TransverseIsotropicMaterial({self.youngs_modulus_parallel}, "
            f"{self.youngs_modulus_transverse}, {self.poisson_ratio_transverse}, "
            f"{self.shear_modulus})"
        )

    def compliance_tensor(self):
        return ComplianceTensor(np.linalg.inv(self.stiffness_tensor().data))

    def stiffness_tensor(self):
        Ex = self.youngs_modulus_parallel
        Eyz = self.youngs_modulus_transverse
        vx = self.poisson_ratio_transverse
        vzy = self.poisson_ratio_transverse
        S_11 = 1 / Ex
        S_12 = -vx / Ex
        S_33 = 1 / Eyz
        S_44 = 1 / (2 * (1 + vzy)) * (Eyz - Ex)
        S_66 = (2 * (1 + vzy)) / Ex

        return StiffnessTensor(
            np.array(
                [
                    [S_11, S_12, S_12, 0, 0, 0],
                    [S_12, S_11, S_12, 0, 0, 0],
                    [S_12, S_12, S_33, 0, 0, 0],
                    [0, 0, 0, S_44, 0, 0],
                    [0, 0, 0, 0, S_44, 0],
                    [0, 0, 0, 0, 0, S_66],
                ]
            )
        )
