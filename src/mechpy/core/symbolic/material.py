import sympy as sp

from .tensor import SymbolicSixBySixTensor


class SymbolicComplianceTensor(SymbolicSixBySixTensor):
    """
    Represents the compliance tensor used in elasticity calculations.

    :param data: The tensor data.
    :param name: The name of the tensor.
    """
    def __init__(self, data, name=None):
        super().__init__(data, name)


class SymbolicStiffnessTensor(SymbolicSixBySixTensor):
    """
    Represents the stiffness tensor used in elasticity calculations.

    :param data: The tensor data.
    :param name: The name of the tensor.
    """
    def __init__(self, data, name=None):
        super().__init__(data, name)


class SymbolicMaterial:
    """
    Represents a general material with symbolic properties.

    :param material_props: Arbitrary keyword arguments representing material properties.
    """
    def __init__(self, **material_props):
        for k, v in material_props.items():
            setattr(self, k, v)

    def __repr__(self):
        """
        Returns a string representation of the material and its properties.

        :return: String representation of the material.
        """
        props = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({props})"


class SymbolicElasticMaterial(SymbolicMaterial):
    """
    Represents an elastic material with methods to compute stiffness and compliance tensors.

    :param material_props: Arbitrary keyword arguments representing material properties.
    """
    def __init__(self, **material_props):
        super().__init__(**material_props)

    def stiffness_tensor(self):
        """
        Computes the stiffness tensor for the material.

        :raises NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def compliance_tensor(self) -> SymbolicComplianceTensor:
        """
        Computes the compliance tensor for the material.

        :return: The computed compliance tensor.
        """
        components = self.stiffness_tensor().to_matrix().inv()
        data = sp.ImmutableDenseNDimArray(components)
        return SymbolicComplianceTensor(sp.simplify(data))


class SymbolicIsotropicMaterial(SymbolicElasticMaterial):
    """
    Represents an isotropic elastic material with properties such as Young's modulus and Poisson's ratio.

    :param material_props: Arbitrary keyword arguments representing material properties.
    """
    def __init__(self, **material_props):
        keys = material_props.keys()
        if not ({"E", "nu"} <= set(keys) or {"lamda", "mu"} <= set(keys)):
            raise ValueError(
                "Material properties must include either both 'E' and 'nu' or both 'lamda' and 'mu'"
            )
        if {"E", "nu"} <= set(keys):
            self.E = material_props.pop("E")
            self.nu = material_props.pop("nu")
        if {"lamda", "mu"} <= set(keys):
            self.lamda = material_props.pop("lamda")
            self.mu = material_props.pop("mu")
        super().__init__(**material_props)

    @staticmethod
    def lame_params(E, nu):
        """
        Converts Young's modulus and Poisson's ratio to Lamé parameters.

        :param E: Young's modulus.
        :param nu: Poisson's ratio.
        :return: Tuple of Lamé parameters (lambda, mu).
        """
        lamda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lamda, mu

    @staticmethod
    def youngs_params(lamda, mu):
        """
        Converts Lamé parameters to Young's modulus and Poisson's ratio.

        :param lamda: First Lamé parameter.
        :param mu: Second Lamé parameter (shear modulus).
        :return: Tuple of Young's modulus and Poisson's ratio (E, nu).
        """
        E = mu * (3 * lamda + 2 * mu) / (lamda + mu)
        mu = lamda / (2 * (lamda + mu))
        return E, mu

    def get_lame_params(self):
        if hasattr(self, "lamda") and hasattr(self, "mu"):
            return self.lamda, self.mu
        if hasattr(self, "E") and hasattr(self, "nu"):
            return self.lame_params(self.E, self.nu)
        raise NotImplementedError()

    def stiffness_tensor(self, lames_param=True) -> SymbolicStiffnessTensor:
        """
        Computes the stiffness tensor for the isotropic material.

        :param lames_param: If True, use Lamé parameters; otherwise, use Young's modulus and Poisson's ratio.
        :return: The computed stiffness tensor.
        """
        if lames_param and hasattr(self, "lamda") and hasattr(self, "mu"):
            lamda, mu = self.lamda, self.mu
        elif hasattr(self, "E") and hasattr(self, "nu"):
            lamda, mu = SymbolicIsotropicMaterial.lame_params(self.E, self.nu)
        else:
            raise AttributeError(
                "Material properties must include either both 'E' and 'nu' or both 'lamda' and 'mu'"
            )

        C_11 = lamda + 2 * mu
        C_12 = lamda
        C_44 = mu
        C = sp.NDimArray(
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
        """
        Computes the compliance tensor for the isotropic material.

        :return: The computed compliance tensor.
        """
        if hasattr(self, "E") and hasattr(self, "nu"):
            E, nu = self.E, self.nu
        elif hasattr(self, "lamda") and hasattr(self, "mu"):
            E, nu = SymbolicIsotropicMaterial.youngs_params(self.lamda, self.mu)
        else:
            raise AttributeError(
                "Material properties must include either both 'E' and 'nu' or both 'lamda' and 'mu'"
            )

        S_11 = 1 / E
        S_12 = -(nu / E)
        S_44 = (2 * (1 + nu)) / E

        S = sp.NDimArray(
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
    """
    Represents a transverse isotropic material with specific elastic properties.

    :param material_props: Arbitrary keyword arguments representing material properties.
    """
    props_keys = {"E_L", "E_T", "nu", "G_L", "G_T"}

    def __init__(self, **material_props):
        keys = material_props.keys()
        if not self.props_keys <= set(keys):
            raise AttributeError(f"Material properties must include {self.props_keys}")
        self.E_L = material_props.pop("E_L")
        self.E_T = material_props.pop("E_T")
        self.nu = material_props.pop("nu")
        self.G_L = material_props.pop("G_L")
        self.G_T = material_props.pop("G_T")
        super().__init__(**material_props)

    def stiffness_tensor(self) -> SymbolicStiffnessTensor:
        """
        Computes the stiffness tensor for the transverse isotropic material.

        :return: The computed stiffness tensor.
        """
        E_L = self.E_L
        E_T = self.E_T
        nu = self.nu
        G_L = self.G_L
        G_T = self.G_T

        C_11 = E_L / (1 - nu**2)
        C_12 = E_L * nu / (1 - nu)
        C_33 = E_T
        C_44 = G_L
        C_66 = G_T

        C = sp.NDimArray(
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
    """
    Represents an orthotropic material with specific elastic properties.

    :param material_props: Arbitrary keyword arguments representing material properties.
    """
    props_keys = {"E1", "E2", "E3", "G12", "G23", "G31", "nu12", "nu23", "nu31"}

    def __init__(self, **material_props):
        keys = material_props.keys()
        if not self.props_keys <= set(keys):
            raise AttributeError(f"Material properties must include {self.props_keys}")
        self.E1 = material_props.pop("E1")
        self.E2 = material_props.pop("E2")
        self.E3 = material_props.pop("E3")
        self.G12 = material_props.pop("G12")
        self.G23 = material_props.pop("G23")
        self.G31 = material_props.pop("G31")
        self.nu12 = material_props.pop("nu12")
        self.nu23 = material_props.pop("nu23")
        self.nu31 = material_props.pop("nu31")
        super().__init__(**material_props)

    def stiffness_tensor(self) -> SymbolicStiffnessTensor:
        """
        Computes the stiffness tensor for the orthotropic material.

        :return: The computed stiffness tensor.
        """
        C11, C22, C33 = self.E1, self.E2, self.E3
        C44, C55, C66 = self.G23, self.G31, self.G12
        C12 = self.nu12 * C22
        C13 = self.nu31 * C11
        C23 = self.nu23 * C33

        C = sp.NDimArray(
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
