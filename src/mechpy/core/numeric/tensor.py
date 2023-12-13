import numpy as np


class Tensor:
    """
    Represents a general tensor object that encapsulates a NumPy array.

    Attributes:
        data (np.ndarray): The data stored in the tensor.

    Methods:
        is_symmetric: Check if the tensor is symmetric.
        is_second_rank: Check if the tensor is of second rank.
        is_fourth_rank: Check if the tensor is of fourth rank.
        is_3x3: Check if the tensor is a 3x3 matrix.
        is_6x6: Check if the tensor is a 6x6 matrix.
        is_3x3x3x3: Check if the tensor is a 3x3x3x3 matrix.
        to_3x3: Convert the tensor to a ThreeByThreeTensor.
        to_sym_3x3: Convert the tensor to a symmetric 3x3 tensor.
        to_6x6: Convert the tensor to a SixBySixTensor.
        from_list: Create a Tensor from a list of components.
    """

    def __init__(self, data):
        """
        Initialize a Tensor object.

        Args:
            data (np.ndarray): A NumPy array representing the tensor data.

        Raises:
            ValueError: If the input data is not a NumPy array.
        """
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError("Input data must be a NumPy array")

    def __repr__(self):
        """
        Return a string representation of the Tensor object.
        """
        return f"Tensor(\n{self.data}\n)"

    def __matmul__(self, other):
        """
        Perform matrix multiplication between two Tensor objects.

        Args:
            other (Tensor): The right operand in the matrix multiplication.

        Returns:
            Tensor: The result of the matrix multiplication.
        """
        # Implementation details...

    def is_symmetric(self):
        """
        Check if the tensor is symmetric.

        Returns:
            bool: True if the tensor is symmetric, False otherwise.
        """
        # if isinstance(self, SymmetricThreeByThreeTensor):
        #     return True
        return np.array_equal(self.data, self.data.T)

    def is_second_rank(self):
        return self.data.ndim == 2

    def is_fourth_rank(self):
        return self.data.ndim == 4

    def is_3x3(self):
        # if isinstance(self, SymmetricThreeByThreeTensor):
        #     return True
        return self.data.shape == (3, 3)

    def is_6x6(self):
        return self.data.shape == (6, 6)

    def is_3x3x3x3(self):
        return self.data.shape == (3, 3, 3, 3)

    def to_3x3(self):
        if self.is_3x3():
            return ThreeByThreeTensor(self.data)
        raise ValueError("The tensor is not a 3x3 Tensor.")

    def to_sym_3x3(self):
        if self.is_3x3() and self.is_symmetric():
            return ThreeByThreeTensor(self.data).to_symmetric()
        raise ValueError("The tensor is not a symetric 3x3 Tensor.")

    def to_6x6(self):
        if self.is_6x6():
            return SixBySixTensor(self.data)
        raise ValueError("The tensor is not a 6x6 Tensor.")

    @classmethod
    def from_list(cls, components, shape, symmetric=False, mode=1):
        if isinstance(components, list):
            if symmetric:
                if shape == (3, 3) and mode == 1:
                    if len(components) != shape[0] * (shape[1] + 1) // 2:
                        raise ValueError("Invalid components for symmetric tensor")
                    components = [float(_) for _ in components]
                    data = np.array(
                        [
                            [components[0], components[5], components[4]],
                            [components[5], components[1], components[3]],
                            [components[4], components[3], components[2]],
                        ]
                    )
                    return cls(data)
                elif len(shape) == 2 and mode == 2:
                    if len(components) != shape[0] * (shape[1] + 1) // 2:
                        raise ValueError("Invalid components for symmetric tensor")
                    data = np.zeros(shape)
                    index = 0
                    for i in range(shape[0]):
                        for j in range(i, shape[1]):
                            data[i, j] = data[j, i] = components[index]
                            index += 1
                    return cls(data)
                else:
                    raise NotImplementedError("Not implemented")
            if isinstance(shape, tuple):
                try:
                    data = np.array(components).reshape(shape)
                except:
                    raise ValueError("Invalid components or shape value")
                return cls(data)

            else:
                raise ValueError("Invalid shape parameter")
        else:
            raise ValueError("Input must be a list")

    def __getitem__(self, key):
        return self.data[key]


class ThreeByThreeTensor(Tensor):
    shape = (3, 3)

    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 3x3 NumPy array")

    def __repr__(self):
        return f"ThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    def to_symmetric(self):
        if self.is_symmetric():
            IVM = SymmetricThreeByThreeTensor.INVERSE_VOIGT_MAPPING
            data = np.array([self.data[IVM[i]] for i in range(6)])
            data = data.reshape((6, 1))  # Ensure the shape is (6, 1)
            return SymmetricThreeByThreeTensor(data)
        raise ValueError(
            "The tensor is not symmetric, and it cannot be converted to a symmetric tensor."
        )


class SixBySixTensor(Tensor):
    shape = (6, 6)

    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 6x6 NumPy array")

    def __repr__(self):
        return f"SixBySixTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)


class SymmetricThreeByThreeTensor(Tensor):
    shape = (6, 1)

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

    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape == self.shape:
            self.data = data
        else:
            raise ValueError("Input data must be a 6x1 NumPy array")

    def __repr__(self):
        return f"SymmetricThreeByThreeTensor(\n{self.data}\n)"

    def is_symmetric(self):
        return True

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    def to_general_tensor(self):
        VM = self.VOIGT_MAPPING
        components = [[self.data[VM[(i, j)]] for i in range(3)] for j in range(3)]
        return ThreeByThreeTensor(np.array(components).reshape((3, 3)))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, tuple) and len(key) == 2:
            return self.to_general_tensor().data[key]
        else:
            raise ValueError("Key must be int or tuple of 2 elements")

    def eigenvalues(self):
        # return np.linalg.eigvals(self.to_general_tensor().data)

        eigenvalues, _ = np.linalg.eigh(self.to_general_tensor().data)
        return eigenvalues
