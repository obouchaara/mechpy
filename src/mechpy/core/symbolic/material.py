import sympy as sp

from .tensor import SymbolicSixBySixTensor


class SymbolicComplianceTensor(SymbolicSixBySixTensor):
    def __init__(self, data):
        super().__init__(data)


class SymbolicStiffnessTensor(SymbolicSixBySixTensor):
    def __init__(self, data):
        super().__init__(data)


class SymbolicMaterial:
    def __init__(self, mechanical_props=None, thermic_props=None):
        self.mechanical_props = mechanical_props or {}
        self.thermic_props = thermic_props or {}

    def __repr__(self):
        return f"SymbolicMaterial(mechanical_props={self.mechanical_props}, thermic_props={self.thermic_props})"


class SymbolicElasticMaterial(SymbolicMaterial):
    def __init__(self, mechanical_props=None, **kwargs):
        if mechanical_props:
            super().__init__(mechanical_props)
        elif hasattr(self, "mechanical_props_keys"):
            keys = self.mechanical_props_keys
            if len(kwargs) == 0:
                for key in keys:
                    setattr(self, key, sp.symbols(key))
            elif len(kwargs) == len(keys) and all(key in kwargs for key in keys):
                for key in keys:
                    setattr(self, key, kwargs[key])
            else:
                raise SyntaxError(f"Invalid arguments: {keys} or none.")
            mechanical_props = {key: getattr(self, key, None) for key in keys}
            super().__init__(mechanical_props)
        else:
            raise SyntaxError(
                "Mechanical properties not provided. You must specify mechanical properties or use predefined keys."
            )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.mechanical_props})"

    def stiffness_tensor(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def compliance_tensor(self) -> SymbolicComplianceTensor:
        stiffness_data = self.stiffness_tensor().data
        compliance_data = stiffness_data.inv()
        return SymbolicComplianceTensor(sp.simplify(compliance_data))


class SymbolicIsotropicMaterial(SymbolicElasticMaterial):
    def __init__(self, youngs_modulus=None, poisson_ratio=None, **kwargs):
        if youngs_modulus and poisson_ratio:
            self.youngs_modulus = youngs_modulus
            self.poisson_ratio = poisson_ratio
        elif "lamda" in kwargs and "mu" in kwargs:
            lamda = kwargs["lamda"]
            mu = kwargs["mu"]
            self.youngs_modulus = mu * (3 * lamda + 2 * mu) / (lamda + mu)
            self.poisson_ratio = lamda / (2 * (lamda + mu))
            self.lamda = lamda
            self.mu = mu
        else:
            E, nu = sp.symbols("E nu")
            self.youngs_modulus = E
            self.poisson_ratio = nu

        mechanical_props = {
            "youngs_modulus": self.youngs_modulus,
            "poisson_ratio": self.poisson_ratio,
        }
        super().__init__(mechanical_props=mechanical_props)

    def __repr__(self):
        return f"SymbolicIsotropicMaterial({self.youngs_modulus}, {self.poisson_ratio})"

    def get_lame_params(self, E, nu):
        lamda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return (lamda, mu)

    def get_youngs_params(self, lamda, mu):
        E = mu * (3 * lamda + 2 * mu) / (lamda + mu)
        mu = lamda / (2 * (lamda + mu))
        return (E, mu)

    def stiffness_tensor(self, lames_param=True) -> SymbolicStiffnessTensor:
        if hasattr(self, "lamda") and hasattr(self, "mu"):
            lamda, mu = self.lamda, self.mu
        else:
            lamda, mu = sp.symbols("lamda mu")
        C_11 = lamda + 2 * mu
        C_12 = lamda
        C_44 = mu

        C = sp.Matrix(
            [
                [C_11, C_12, C_12, 0, 0, 0],
                [C_12, C_11, C_12, 0, 0, 0],
                [C_12, C_12, C_11, 0, 0, 0],
                [0, 0, 0, C_44, 0, 0],
                [0, 0, 0, 0, C_44, 0],
                [0, 0, 0, 0, 0, C_44],
            ]
        )

        if not lames_param:
            E = self.youngs_modulus
            nu = self.poisson_ratio
            lamda_expr, mu_expr = self.get_lame_params(E, nu)
            C = C.subs({lamda: lamda_expr, mu: mu_expr})

        return SymbolicStiffnessTensor(sp.simplify(C))

    def compliance_tensor(self) -> SymbolicComplianceTensor:
        E = self.youngs_modulus
        nu = self.poisson_ratio
        S_11 = 1 / E
        S_12 = -(nu / E)
        S_44 = (2 * (1 + nu)) / E

        S = sp.Matrix(
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
    def __init__(
        self,
        youngs_modulus_parallel=None,
        youngs_modulus_transverse=None,
        poisson_ratio=None,
        shear_modulus_parallel=None,
        shear_modulus_transverse=None,
    ):
        if (
            youngs_modulus_parallel
            and youngs_modulus_transverse
            and poisson_ratio
            and shear_modulus_parallel
            and shear_modulus_transverse
        ):
            self.youngs_modulus_parallel = youngs_modulus_parallel
            self.youngs_modulus_transverse = youngs_modulus_transverse
            self.poisson_ratio = poisson_ratio
            self.shear_modulus_parallel = shear_modulus_parallel
            self.shear_modulus_transverse = shear_modulus_transverse
        else:
            E_L, E_T, nu, G_L, G_T = sp.symbols("E_L E_T nu G_L G_T")
            self.youngs_modulus_parallel = E_L
            self.youngs_modulus_transverse = E_T
            self.poisson_ratio = nu
            self.shear_modulus_parallel = G_L
            self.shear_modulus_transverse = G_T

        mechanical_props = {
            "youngs_modulus_parallel": self.youngs_modulus_parallel,
            "youngs_modulus_transverse": self.youngs_modulus_transverse,
            "poisson_ratio": self.poisson_ratio,
            "shear_modulus_parallel": self.shear_modulus_parallel,
            "shear_modulus_transverse": self.shear_modulus_transverse,
        }

        super().__init__(mechanical_props=mechanical_props)

    def __repr__(self):
        return (
            f"TransverseIsotropicMaterial({self.youngs_modulus_parallel}, "
            f"{self.youngs_modulus_transverse}, {self.poisson_ratio}, "
            f"{self.shear_modulus_parallel}, {self.shear_modulus_transverse})"
        )

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

        C = sp.Matrix(
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
    mechanical_props_keys = [
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
        super().__init__(**kwargs)

    def stiffness_tensor(self) -> SymbolicStiffnessTensor:
        C11, C22, C33 = self.E1, self.E2, self.E3
        C44, C55, C66 = self.G23, self.G31, self.G12
        C12 = self.nu12 * C22
        C13 = self.nu31 * C11
        C23 = self.nu23 * C33

        C = sp.Matrix(
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
