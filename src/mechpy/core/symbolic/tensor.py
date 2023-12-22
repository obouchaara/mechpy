"""
SymbolicTensor Module
=====================

This module offers classes for symbolic tensor manipulation using SymPy. It's designed to handle various tensor operations in a symbolic manner, ideal for applications in physics, engineering, and related fields.

Classes:
    `SymbolicTensor <#mechpy.core.symbolic.tensor.SymbolicTensor>`_:
        A base class for symbolic tensors. It handles initialization with SymPy Array objects and provides basic tensor properties and operations, including checks for symmetry and shape.

    `SymbolicThreeByThreeTensor <#mechpy.core.symbolic.tensor.SymbolicThreeByThreeTensor>`_:
        Inherits from SymbolicTensor, specifically representing a 3x3 symbolic tensor. Offers additional methods and properties tailored to the 3x3 tensor structure.

    `SymbolicSixBySixTensor <#mechpy.core.symbolic.tensor.SymbolicSixBySixTensor>`_:
        Extends SymbolicTensor for 6x6 symbolic tensors. Includes specific functionalities for working with 6x6 tensor.

    `SymbolicSymmetricThreeByThreeTensor <#mechpy.core.symbolic.tensor.SymbolicSymmetricThreeByThreeTensor>`_:
        A specialization for 3x3 symmetric tensors. It incorporates Voigt mapping for tensor transformations and eigenvalue computations, focusing on symmetric properties.

The module emphasizes ease of use and flexibility in symbolic tensor computations, leveraging the power of SymPy for mathematical expressions.
"""

import itertools
import sympy as sp


class SymbolicTensor:
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

    def __init__(self, data):
        if isinstance(data, sp.ImmutableDenseNDimArray):
            self.data = data
        else:
            raise ValueError("Input data must be a SymPy ImmutableDenseNDimArray")

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data}\n)"

    def is_second_rank(self):
        return len(self.data.shape) == 2

    def is_fourth_rank(self):
        return len(self.data.shape) == 4

    def is_square(self):
        return self.is_second_rank() and self.data.shape[0] == self.data.shape[1]

    def is_symmetric(self):
        return self.is_square() and self.data == self.data.transpose()

    def to_matrix(self):
        if self.is_square():
            return sp.Matrix(self.data.tolist())
        raise ValueError("The Tensor should be a second rank square")

    def to_3x3(self):
        if self.data.shape == (3, 3):
            return SymbolicThreeByThreeTensor(self.data)
        raise ValueError("The tensor is not a 3x3 Array.")

    def to_sym_3x3(self, notation=1):
        if self.data.shape == (3, 3) and self.is_symmetric():
            return self.to_3x3().to_symmetric(notation)
        raise ValueError("The tensor is not a symmetric 3x3 Array.")

    def to_6x6(self):
        if self.data.shape == (6, 6):
            return SymbolicSixBySixTensor(self.data)
        raise ValueError("The tensor is not a 6x6 Array.")

    @classmethod
    def from_list(cls, components, shape):
        if isinstance(components, list):
            if isinstance(shape, tuple):
                try:
                    data = sp.ImmutableDenseNDimArray(components, shape)
                    return cls(data)
                except:
                    raise ValueError("Invalid components or shape value")
            raise ValueError("Invalid shape parameter")
        raise ValueError("Input must be a list")

    @classmethod
    def create(cls, name, shape):
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple")

        indices = itertools.product(*(range(1, dim_len + 1) for dim_len in shape))
        components = [sp.symbols(f"{name}_{''.join(map(str, idx))}") for idx in indices]
        data = sp.ImmutableDenseNDimArray(components, shape)
        return cls(data)

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
        result_data = sp.ImmutableDenseNDimArray(self.to_matrix() @ other.to_matrix())
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

    def __init__(self, data):
        if isinstance(data, sp.MutableDenseNDimArray):
            data = sp.ImmutableDenseNDimArray(data)
        if isinstance(data, sp.ImmutableDenseNDimArray) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 3x3 SymPy Array")

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    @classmethod
    def create(cls, name):
        return super().create(name, cls.shape)

    def to_symmetric(self, notation=1):
        if self.is_symmetric():
            NOTATIONS = SymbolicSymmetricThreeByThreeTensor.NOTATIONS
            if notation in NOTATIONS.keys():
                mapping = NOTATIONS[notation]["inverse_map"]
                components = [self.data[mapping[i]] for i in range(6)]
                data = sp.ImmutableDenseNDimArray(components)
                return SymbolicSymmetricThreeByThreeTensor(data, notation)
            raise NotImplementedError(f"Notation {notation} not implemented")
        raise ValueError("The tensor is not symmetric")


class SymbolicSixBySixTensor(SymbolicTensor):
    shape = (6, 6)

    def __init__(self, data):
        if isinstance(data, sp.MutableDenseNDimArray):
            data = sp.ImmutableDenseNDimArray(data)
        if isinstance(data, sp.ImmutableDenseNDimArray) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 6x6 SymPy Array")

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    @classmethod
    def create(cls, name):
        return super().create(name, cls.shape)


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
        1: {
            "name": "Standard notation",
            "map": STANDARD_MAPPING,
            "inverse_map": INVERSE_STANDARD_MAPPING,
        },
        2: {
            "name": "Voight notation",
            "map": VOIGT_MAPPING,
            "inverse_map": INVERSE_VOIGT_MAPPING,
        },
    }

    def __init__(self, data, notation=1):
        if isinstance(data, sp.MutableDenseNDimArray):
            data = sp.ImmutableDenseNDimArray(data)
        if isinstance(data, sp.ImmutableDenseNDimArray) and data.shape == self.shape:
            if notation in self.NOTATIONS.keys():
                super().__init__(data)
                self.notation = notation
            else:
                raise NotImplementedError(f"Notation {notation} not implemented")
        else:
            raise ValueError(f"Input data must be a SymPy Array ans sahpe={self.shape}")

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data},\n{self.notation}\n)"

    def is_symmetric(self):
        return True

    @classmethod
    def from_list(cls, components, notation=1):
        if isinstance(components, list):
            if len(components) == 6:
                if notation in cls.NOTATIONS.keys():
                    data = sp.ImmutableDenseNDimArray(components)
                    return cls(data, notation)
                raise NotImplementedError(f"Notation {notation} not implemented")
            elif len(components) == 9:
                return SymbolicThreeByThreeTensor.from_list(components).to_symmetric(
                    notation
                )
            raise ValueError("Imput must be a list of 6 or 9 elements")
        raise ValueError("Input must be a list")

    @classmethod
    def create(cls, name, notation=1):
        if notation == 1:
            return super().create(name, cls.shape)
        elif notation == 2:
            mapping = cls.INVERSE_VOIGT_MAPPING
            components = [sp.symbols(f"{name}_{i+1}{j+1}") for i, j in mapping.values()]
            return cls.from_list(components, notation)
        else:
            raise NotImplementedError(f"Notation {notation} not implemented")

    def to_general(self):
        mapping = self.NOTATIONS[self.notation]["map"]
        components = [[self.data[mapping[(i, j)]] for j in range(3)] for i in range(3)]
        data = sp.ImmutableDenseNDimArray(components)
        return SymbolicThreeByThreeTensor(data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, tuple) and len(key) == 2:
            return self.data[self.VOIGT_MAPPING[key]]
        else:
            raise ValueError("Key must be int or tuple of 2 elements")

    def eigenvalues(self):
        return self.to_general().eigenvalues()

    def eigenvectors(self):
        return self.to_general().eigenvectors()

    def diagonalize(self):
        return self.to_general().diagonalize()
