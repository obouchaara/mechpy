"""
SymbolicTensor Module
=====================

This module offers classes for symbolic tensor manipulation using SymPy. It's designed to handle various tensor operations in a symbolic manner, ideal for applications in physics, engineering, and related fields.

Classes:
    SymbolicTensor:
        A base class for symbolic tensors. It handles initialization with SymPy Array objects and provides basic tensor properties and operations, including checks for symmetry and shape.

    SymbolicThreeByThreeTensor:
        Inherits from SymbolicTensor, specifically representing a 3x3 symbolic tensor. Offers additional methods and properties tailored to the 3x3 tensor structure.

    SymbolicSixBySixTensor:
        Extends SymbolicTensor for 6x6 symbolic tensors. Includes specific functionalities for working with 6x6 tensor.

    SymbolicSymmetricThreeByThreeTensor:
        A specialization for 3x3 symmetric tensors. It incorporates Voigt mapping for tensor transformations and eigenvalue computations, focusing on symmetric properties.

The module emphasizes ease of use and flexibility in symbolic tensor computations, leveraging the power of SymPy for mathematical expressions.
"""

import itertools
import sympy as sp


class SymbolicTensor:
    def __init__(self, data, name=None, tensor_params=None):
        if not isinstance(data, sp.NDimArray):
            raise ValueError("Input data must be a SymPy NDimArray.")

        self.data = data
        self.name = name
        self.tensor_params = tensor_params or {}

        if not isinstance(self.tensor_params, dict):
            raise ValueError("Tensor parameters must be a dict.")

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data},\n{self.name}\n)"

    @staticmethod
    def notation_standard_map(n):
        """
        Create a dictionary with tuple (i, j) as key and index in the flattened list as value.
        The dictionary maps the position in an nxn symmetric matrix to its corresponding index in a flattened list.

        :param n: Size of the matrix (nxn).
        :return: Dictionary mapping (i, j) to index.
        """
        index_map = {}
        index = 0
        for i in range(n):
            for j in range(i, n):
                index_map[(i, j)] = index
                index_map[(j, i)] = index  # Mirror for symmetry
                index += 1

        return index_map

    @staticmethod
    def notation_standard_inverse_map(n):
        """
        Create a dictionary with index in the flattened list as key and tuple (i, j) as value.
        The dictionary inversely maps the index in a flattened list to its corresponding position in an nxn symmetric matrix.

        :param n: Size of the matrix (nxn).
        :return: Dictionary mapping index to (i, j).
        """
        index_map_inverse = {}
        index = 0
        for i in range(n):
            for j in range(i, n):
                index_map_inverse[index] = (i, j)
                index += 1

        return index_map_inverse

    def is_second_rank(self):
        return len(self.data.shape) == 2

    def is_fourth_rank(self):
        return len(self.data.shape) == 4

    def is_square(self):
        return self.is_second_rank() and self.data.shape[0] == self.data.shape[1]

    def is_symmetric(self):
        return self.is_square() and self.data == self.data.transpose()

    def to_matrix(self):
        if not self.is_square():
            raise ValueError("The Tensor should be a second rank square.")
        return sp.MutableDenseMatrix(self.data.tolist())

    def to_3x3(self):
        if self.data.shape == (3, 3):
            return SymbolicThreeByThreeTensor(
                data=self.data,
                name=self.name,
                tensor_params=self.tensor_params,
            )
        raise ValueError("The tensor is not a 3x3 Array.")

    def to_sym_3x3(self, notation=None):
        if notation is None:
            notation = "standard"
        if self.data.shape == (3, 3) and self.is_symmetric():
            return self.to_3x3().to_symmetric(notation)
        raise ValueError("The tensor is not a symmetric 3x3 Array.")

    def to_6x6(self):
        if self.data.shape == (6, 6):
            return SymbolicSixBySixTensor(self.data)
        raise ValueError("The tensor is not a 6x6 Array.")

    @classmethod
    def from_list(cls, components, shape):
        if not isinstance(components, list):
            raise ValueError("Components must be a list.")
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple.")

        # need components length validation

        data = sp.NDimArray(components, shape=shape)
        return cls(data)

    @classmethod
    def create(cls, shape, name):
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple.")
        if name is None:
            raise ValueError("Name cannot be None.")

        indices = itertools.product(*(range(1, dim_len + 1) for dim_len in shape))
        components = [sp.symbols(f"{name}_{''.join(map(str, idx))}") for idx in indices]
        data = sp.NDimArray(components, shape)
        return cls(data, name=name)

    def subs_tensor_params(self, param_values):
        if not isinstance(param_values, dict):
            raise TypeError("param_values must be a dictionary")

        # Perform the substitution for provided parameters
        for param, value in param_values.items():
            if not param in self.tensor_params:
                raise ValueError(f"Parameter {param} not found in tensor parameters")

            self.data = self.data.subs(param, value)
            self.tensor_params.pop(param)

    def subs_tensor_components(self, components_values):
        data = sp.MutableDenseNDimArray(self.data)
        for key, value in components_values.items():
            data[key] = value
        self.data = sp.NDimArray(data)

    def subs(self, subs_dict):
        try:
            self.data = self.data.subs(subs_dict)
        except Exception as e:
            raise RuntimeError(f"An error occurred during substitution: {e}")

    def __matmul__(self, other):
        if not isinstance(other, SymbolicTensor):
            raise ValueError("The other operand must be an instance of SymbolicTensor")

        if not (self.is_second_rank() and other.is_second_rank()):
            raise ValueError("Requires second rank tensors")

        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(
                "Shapes are not aligned for matrix multiplication: "
                f"{self.data.shape} and {other.data.shape}"
            )
        result_data = sp.NDimArray(self.to_matrix() @ other.to_matrix())
        return SymbolicTensor(result_data)

    def __getitem__(self, key):
        return self.data[key]

    def eigenvalues(self):
        return self.to_matrix().eigenvals()

    def eigenvectors(self):
        return self.to_matrix().eigenvects()

    def diagonalize(self):
        return self.to_matrix().diagonalize()


class SymbolicThreeByThreeTensor(SymbolicTensor):
    shape = (3, 3)

    def __init__(self, data, name=None, tensor_params=None):
        if not isinstance(data, sp.NDimArray) or data.shape != self.shape:
            raise ValueError("Data must be a 3x3 SymPy Array.")
        super().__init__(data, name, tensor_params)

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    @classmethod
    def create(cls, name):
        return super().create(cls.shape, name)

    def to_symmetric(self, notation=None):
        if notation is None:
            notation = "standard"
        if not self.is_symmetric():
            raise ValueError("The tensor is not symmetric.")
        NOTATIONS = SymbolicSymmetricThreeByThreeTensor.NOTATIONS
        if notation not in NOTATIONS.keys():
            raise NotImplementedError(f"Notation {notation} not implemented")

        mapping = NOTATIONS[notation]["inverse_map"]
        components = [self.data[mapping[i]] for i in range(6)]
        data = sp.NDimArray(components)
        return SymbolicSymmetricThreeByThreeTensor(data, notation=notation)


class SymbolicSixBySixTensor(SymbolicTensor):
    shape = (6, 6)

    def __init__(self, data, name=None, tensor_params=None):
        if not isinstance(data, sp.NDimArray) or data.shape != self.shape:
            raise ValueError("Data must be a 6x6 SymPy Array.")
        super().__init__(data, name, tensor_params)

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    @classmethod
    def create(cls, name):
        return super().create(cls.shape, name=name)


class SymbolicSymmetricThreeByThreeTensor(SymbolicTensor):
    shape = (6,)

    STANDARD_MAPPING = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 1,
        (1, 1): 3,
        (1, 2): 4,
        (2, 0): 2,
        (2, 1): 4,
        (2, 2): 5,
    }

    INVERSE_STANDARD_MAPPING = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (1, 1),
        4: (1, 2),
        5: (2, 2),
    }

    VOIGT_MAPPING = {
        (0, 0): 0,
        (1, 1): 1,
        (2, 2): 2,
        (1, 2): 3,
        (2, 1): 3,
        (0, 2): 4,
        (2, 0): 4,
        (0, 1): 5,
        (1, 0): 5,
    }

    INVERSE_VOIGT_MAPPING = {
        0: (0, 0),
        1: (1, 1),
        2: (2, 2),
        3: (1, 2),
        4: (0, 2),
        5: (0, 1),
    }

    NOTATIONS = {
        "standard": {
            "name": "Standard notation",
            "map": STANDARD_MAPPING,
            "inverse_map": INVERSE_STANDARD_MAPPING,
        },
        "voigt": {
            "name": "Voigt notation",
            "map": VOIGT_MAPPING,
            "inverse_map": INVERSE_VOIGT_MAPPING,
        },
        1: {
            "name": "Standard notation",
            "map": STANDARD_MAPPING,
            "inverse_map": INVERSE_STANDARD_MAPPING,
        },
        2: {
            "map": VOIGT_MAPPING,
            "inverse_map": INVERSE_VOIGT_MAPPING,
        },
    }

    def __init__(
        self,
        data,
        name=None,
        notation=None,
        tensor_params=None,
    ):
        if not isinstance(data, sp.NDimArray) or data.shape != self.shape:
            raise ValueError("Data must be a 6x1 SymPy Array.")
        if notation is None:
            notation = "standard"
        if notation not in self.NOTATIONS.keys():
            raise NotImplementedError(f"Notation {notation} not implemented.")

        super().__init__(data, name, tensor_params)
        self.notation = notation

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data},\n{self.name},\n{self.notation}\n)"

    def is_symmetric(self):
        return True

    @classmethod
    def from_list(
        cls,
        components,
        name=None,
        notation=None,
    ):
        if not isinstance(components, list):
            raise ValueError("Input must be a list")
        if notation is None:
            notation = "standard"
        if notation not in cls.NOTATIONS.keys():
            raise NotImplementedError(f"Notation {notation} not implemented")

        if len(components) == 6:
            data = sp.NDimArray(components)
            return cls(data, name=name, notation=notation)
        elif len(components) == 9:
            tensor = SymbolicThreeByThreeTensor.from_list(components)
            return tensor.to_symmetric(notation)
        else:
            raise ValueError("Imput must be a list of 6 or 9 elements")

    @classmethod
    def create(
        cls,
        name,
        notation=None,
    ):
        if notation is None:
            notation = "standard"
        if notation == "voigt":
            mapping = cls.INVERSE_VOIGT_MAPPING
            components = [sp.symbols(f"{name}_{i+1}{j+1}") for i, j in mapping.values()]
            return cls.from_list(components, name=name, notation=notation)
        return super().create(cls.shape, name)

    def to_general(self):
        mapping = self.NOTATIONS[self.notation]["map"]
        components = [[self.data[mapping[(i, j)]] for j in range(3)] for i in range(3)]
        data = sp.NDimArray(components)
        return SymbolicThreeByThreeTensor(data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, tuple) and len(key) == 2:
            return self.data[self.NOTATIONS[self.notation]["map"][key]]
        else:
            raise ValueError("Key must be int or tuple of 2 elements")

    def eigenvalues(self):
        return self.to_general().eigenvalues()

    def eigenvectors(self):
        return self.to_general().eigenvectors()

    def diagonalize(self):
        return self.to_general().diagonalize()
