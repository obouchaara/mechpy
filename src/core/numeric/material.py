import numpy as np

from .tensor import SixBySixTensor


class AnisotropicMaterial:
    def __init__(self):
        pass

    def __repr__(self):
        return f"AnisotropicMaterial()"


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


class IsotropicMaterial(AnisotropicMaterial):
    def __init__(self, youngs_modulus, poisson_ratio, lames_lambda=None, lames_mu=None):
        if youngs_modulus and poisson_ratio:
            self.youngs_modulus = youngs_modulus
            self.poisson_ratio = poisson_ratio

            self.lames_lambda = youngs_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
            self.lames_mu = youngs_modulus / 2 / (1 + poisson_ratio)

        elif lames_lambda and lames_mu:
            self.youngs_modulus = 2 * lames_mu * (1 + poisson_ratio)
            self.poisson_ratio = (2 * lames_mu + 3 * lames_lambda) / (
                2 * lames_mu + 2 * lames_lambda
            )

            self.lames_lambda = lames_lambda
            self.lames_mu = lames_mu

        else:
            raise ValueError(
                "Either Young's modulus and Poisson's ratio or Lame's lambda and Lame's mu must be provided."
            )

    def __repr__(self):
        return f"IsotropicMaterial({self.youngs_modulus}, {self.poisson_ratio}, {self.lames_lambda}, {self.lames_mu})"

    def compliance_tensor(self):
        C_11 = self.lames_lambda + 2 * self.lames_mu
        C_12 = self.lames_lambda
        C_44 = self.lames_mu

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
        S_11 = 1 / self.youngs_modulus
        S_12 = -(self.poisson_ratio / self.youngs_modulus)
        S_44 = (2 * (1 + self.poisson_ratio)) / self.youngs_modulus

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


class TransverseIsotropicMaterial(AnisotropicMaterial):
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

    def __repr__(self):
        return (
            f"TransverseIsotropicMaterial({self.youngs_modulus_parallel}, "
            f"{self.youngs_modulus_transverse}, {self.poisson_ratio_transverse}, "
            f"{self.shear_modulus})"
        )

    def compliance_tensor(self):
        return ComplianceTensor(np.linalg.inv(self.stiffness_tensor().data))

    def stiffness_tensor(self):
        S_11 = 1 / self.youngs_modulus_parallel
        S_12 = -self.poisson_ratio_transverse / self.youngs_modulus_parallel
        S_33 = 1 / self.youngs_modulus_transverse
        S_44 = (
            1
            / (2 * (1 + self.poisson_ratio_transverse))
            * (self.youngs_modulus_transverse - self.youngs_modulus_parallel)
        )
        S_66 = (2 * (1 + self.poisson_ratio_transverse)) / self.youngs_modulus_parallel

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
