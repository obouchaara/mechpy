import sympy as sp

from .tensor import SymbolicSixBySixTensor


class SymbolicComplianceTensor(SymbolicSixBySixTensor):
    def __init__(self, data, name=None):
        super().__init__(data, name)


class SymbolicStiffnessTensor(SymbolicSixBySixTensor):
    def __init__(self, data, name=None):
        super().__init__(data, name)


class SymbolicMaterial:
    def __init__(self, mechanical_props=None, thermic_props=None):
        self.mechanical_props = mechanical_props or {}
        self.thermic_props = thermic_props or {}

    def __repr__(self):
        return f"{self.__class__.__name__}(\nmechanical_props={self.mechanical_props},\nthermic_props={self.thermic_props}\n)"


class SymbolicElasticMaterial(SymbolicMaterial):
    def __init__(self, mechanical_props=None):
        super().__init__(mechanical_props)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.mechanical_props}\n)"

    def stiffness_tensor(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def compliance_tensor(self) -> SymbolicComplianceTensor:
        components = self.stiffness_tensor().to_matrix().inv()
        data = sp.ImmutableDenseNDimArray(components)
        return SymbolicComplianceTensor(sp.simplify(data))


class SymbolicIsotropicMaterial(SymbolicElasticMaterial):
    def __init__(self, youngs_modulus=None, poisson_ratio=None, **kwargs):
        if youngs_modulus and poisson_ratio:
            self.youngs_modulus = youngs_modulus
            self.poisson_ratio = poisson_ratio
        elif "lamda" in kwargs and "mu" in kwargs:
            lamda = kwargs["lamda"]
            mu = kwargs["mu"]
            self.lamda = lamda
            self.mu = mu
        else:
            E, nu = sp.symbols("E nu")
            self.youngs_modulus = E
            self.poisson_ratio = nu

        if hasattr(self, "youngs_modulus") and hasattr(self, "poisson_ratio"):
            mechanical_props = {
                "youngs_modulus": self.youngs_modulus,
                "poisson_ratio": self.poisson_ratio,
            }
        elif hasattr(self, "lamda") and hasattr(self, "mu"):
            lamda, mu = self.lamda, self.mu
            E_expr, nu_expr = self.get_youngs_params(lamda, mu)
            mechanical_props = {
                "youngs_modulus": E_expr,
                "poisson_ratio": nu_expr,
            }
        else:
            raise ValueError(
                "Isotropic Material should be defined by (E, nu) or (lamda, mu)"
            )

        super().__init__(mechanical_props=mechanical_props)

    def get_lame_params(self, E, nu):
        lamda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return (lamda, mu)

    def get_youngs_params(self, lamda, mu):
        E = mu * (3 * lamda + 2 * mu) / (lamda + mu)
        mu = lamda / (2 * (lamda + mu))
        return (E, mu)

    def stiffness_tensor(self, lames_param=True) -> SymbolicStiffnessTensor:
        if lames_param:
            if hasattr(self, "lamda") and hasattr(self, "mu"):
                lamda, mu = self.lamda, self.mu
            else:
                lamda, mu = sp.symbols("lamda mu")
        else:
            if hasattr(self, "youngs_modulus") and hasattr(self, "poisson_ratio"):
                E = self.youngs_modulus
                nu = self.poisson_ratio
                lamda, mu = self.get_lame_params(E, nu)
            else:
                raise ValueError(
                    "If lames_param is False the material should be defined by (E, nu)"
                )

        C_11 = lamda + 2 * mu
        C_12 = lamda
        C_44 = mu

        C = sp.ImmutableDenseNDimArray(
            [
                [C_11, C_12, C_12, 0, 0, 0],
                [C_12, C_11, C_12, 0, 0, 0],
                [C_12, C_12, C_11, 0, 0, 0],
                [0, 0, 0, C_44, 0, 0],
                [0, 0, 0, 0, C_44, 0],
                [0, 0, 0, 0, 0, C_44],
            ]
        )

        return SymbolicStiffnessTensor(sp.simplify(C))

    def compliance_tensor(self) -> SymbolicComplianceTensor:
        E = self.mechanical_props["youngs_modulus"]
        nu = self.mechanical_props["poisson_ratio"]
        S_11 = 1 / E
        S_12 = -(nu / E)
        S_44 = (2 * (1 + nu)) / E

        S = sp.ImmutableDenseNDimArray(
            [
                [S_11, S_12, S_12, 0, 0, 0],
                [S_12, S_11, S_12, 0, 0, 0],
                [S_12, S_12, S_11, 0, 0, 0],
                [0, 0, 0, S_44, 0, 0],
                [0, 0, 0, 0, S_44, 0],
                [0, 0, 0, 0, 0, S_44],
            ]
        )

        return SymbolicComplianceTensor(sp.simplify(S))


class SymbolicTransverseIsotropicMaterial(SymbolicElasticMaterial):
    props_dict = {
        "youngs_modulus_parallel": "E_L",
        "youngs_modulus_transverse": "E_T",
        "poisson_ratio": "nu",
        "shear_modulus_parallel": "G_L",
        "shear_modulus_transverse": "G_T",
    }

    def __init__(self, **kwargs):
        mechanical_props = {}

        for prop, symbol_name in self.props_dict.items():
            if prop in kwargs:
                value = kwargs[prop]
            else:
                value = sp.symbols(symbol_name)

            setattr(self, prop, value)
            mechanical_props[prop] = value

        super().__init__(mechanical_props=mechanical_props)

    def stiffness_tensor(self) -> SymbolicStiffnessTensor:
        E_L = self.youngs_modulus_parallel
        E_T = self.youngs_modulus_transverse
        nu = self.poisson_ratio
        G_L = self.shear_modulus_parallel
        G_T = self.shear_modulus_transverse

        C_11 = E_L / (1 - nu**2)
        C_12 = E_L * nu / (1 - nu)
        C_33 = E_T
        C_44 = G_L
        C_66 = G_T

        C = sp.ImmutableDenseNDimArray(
            [
                [C_11, C_12, C_12, 0, 0, 0],
                [C_12, C_11, C_12, 0, 0, 0],
                [C_12, C_12, C_33, 0, 0, 0],
                [0, 0, 0, C_44, 0, 0],
                [0, 0, 0, 0, C_44, 0],
                [0, 0, 0, 0, 0, C_66],
            ]
        )

        return SymbolicStiffnessTensor(sp.simplify(C))


class SymbolicOrthotropicMaterial(SymbolicElasticMaterial):
    props_keys = [
        "E1",
        "E2",
        "E3",
        "G12",
        "G23",
        "G31",
        "nu12",
        "nu23",
        "nu31",
    ]

    def __init__(self, **kwargs):
        mechanical_props = {}

        for prop in self.props_keys:
            if prop in kwargs:
                value = kwargs[prop]
            else:
                value = sp.symbols(prop)

            setattr(self, prop, value)
            mechanical_props[prop] = value

        super().__init__(mechanical_props=mechanical_props)

    def stiffness_tensor(self) -> SymbolicStiffnessTensor:
        C11, C22, C33 = self.E1, self.E2, self.E3
        C44, C55, C66 = self.G23, self.G31, self.G12
        C12 = self.nu12 * C22
        C13 = self.nu31 * C11
        C23 = self.nu23 * C33

        C = sp.ImmutableDenseNDimArray(
            [
                [C11, C12, C13, 0, 0, 0],
                [C12, C22, C23, 0, 0, 0],
                [C13, C23, C33, 0, 0, 0],
                [0, 0, 0, C44, 0, 0],
                [0, 0, 0, 0, C55, 0],
                [0, 0, 0, 0, 0, C66],
            ]
        )

        return SymbolicStiffnessTensor(sp.simplify(C))
