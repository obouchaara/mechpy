import numpy as np
import sympy as sp


class ThreeByThreeTensor:
    def __init__(self, data):
        # Ensure that data is a 3x3 array
        if not isinstance(data, np.ndarray) or data.shape != (3, 3):
            raise ValueError("Input data must be a 3x3 NumPy array")

        self.data = data

    def __repr__(self):
        return f"ThreeByThreeTensor(\n{self.data}\n)"

    def __add__(self, other):
        pass

    def __mul__(self, scalar):
        pass

    @classmethod
    def create(cls, components):
        if isinstance(components, list):
            if len(components) == 9:
                # Create a 3x3 tensor from a flat list of 9 components
                data = np.array(components).reshape((3, 3))
                return cls(data)
            elif len(components) == 3 and all(
                isinstance(row, list) and len(row) == 3 for row in components
            ):
                # Create a 3x3 tensor from a list of 3 lists
                data = np.array(components)
                return cls(data)
            else:
                raise ValueError("Invalid input for creating a 3x3 tensor")
        else:
            raise ValueError("Input must be a list")


class SymmetricThreeByThreeTensor(ThreeByThreeTensor):
    def __init__(self, data):
        # Ensure that the tensor is symmetric
        if not np.allclose(data, data.T):
            raise ValueError("Input data must be a symmetric 3x3 tensor")

        # Call the constructor of the base class
        super().__init__(data)

    def __repr__(self):
        return f"SymmetricThreeByThreeTensor(\n{self.data}\n)"
    
    @classmethod
    def create(cls, components):
        if isinstance(components, list) and len(components) == 6:
            # Create a 3x3 symmetric tensor from a flat list of 6 components
            data = np.array([[components[0], components[1], components[2]],
                              [components[1], components[3], components[4]],
                              [components[2], components[4], components[5]]])
            return cls(data)
        else:
            raise ValueError("Invalid input for creating a symmetric 3x3 tensor. Provide a list of 6 components.")


class SymbolicThreeByThreeTensor:
    def __init__(self, data: sp.Matrix):
        # Ensure that data is a 3x3 matrix
        if not isinstance(data, sp.Matrix) or data.shape != (3, 3):
            raise ValueError("Input data must be a 3x3 SymPy Matrix")

        self.data = data

    def __repr__(self):
        return f"SymbolicThreeByThreeTensor(\n{self.data}\n)"

    def __add__(self, other):
        pass

    def __mul__(self, scalar):
        pass

    @classmethod
    def create(cls, components):
        if isinstance(components, list):
            if len(components) == 9:
                # Create a 3x3 tensor from a flat list of 9 components
                data = sp.Matrix(components).reshape(3, 3)
                return cls(data)
            elif len(components) == 3 and all(
                isinstance(row, list) and len(row) == 3 for row in components
            ):
                # Create a 3x3 tensor from a list of 3 lists
                data = sp.Matrix(components)
                return cls(data)
            else:
                raise ValueError("Invalid input for creating a 3x3 tensor")
        else:
            raise ValueError("Input must be a list")


class SymmetricSymbolicThreeByThreeTensor(SymbolicThreeByThreeTensor):
    def __init__(self, data: sp.Matrix):
        # Ensure that the tensor is symmetric
        if not data.is_symmetric():
            raise ValueError("Input data must be a symmetric 3x3 tensor")

        # Call the constructor of the base class
        super().__init__(data)

    @classmethod
    def create(cls, components):
        if isinstance(components, list) and len(components) == 6:
            # Create a 3x3 symmetric tensor from a flat list of 6 symbolic components
            data = sp.Matrix([[components[0], components[1], components[2]],
                              [components[1], components[3], components[4]],
                              [components[2], components[4], components[5]]])
            return cls(data)
        else:
            raise ValueError("Invalid input for creating a symbolic symmetric 3x3 tensor. Provide a list of 6 symbolic components.")


if __name__ == "__main__":
    tensor1 = ThreeByThreeTensor.create([1, 2, 3, 4, 5, 6, 7, 8, 9])
    tensor2 = ThreeByThreeTensor.create([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print("Tensor 1:")
    print(tensor1)
    print("\n")
    print("Tensor 2:")
    print(tensor2)
    print("\n")


    symmetric_tensor = SymmetricThreeByThreeTensor.create([1, 2, 3, 4, 5, 6])

    print("Symmetric Tensor:")
    print(symmetric_tensor)
    print("\n")



    a, b, c, d, e, f, g, h, i = sp.symbols("a b c d e f g h i")

    symbolic_tensor1 = SymbolicThreeByThreeTensor.create([a, b, c, d, e, f, g, h, i])

    symbolic_tensor2 = SymbolicThreeByThreeTensor.create(
        [
            [a, b, c],
            [d, e, f],
            [g, h, i],
        ]
    )

    print("Symbolic Tensor 1:")
    print(symbolic_tensor1)
    print("\n")
    print("Symbolic Tensor 2:")
    print(symbolic_tensor2)
    print("\n")

    symbolic_symmetric_tensor = SymmetricSymbolicThreeByThreeTensor.create([a, b, c, d, e, f])

    print("Symbolic Symmetric Tensor:")
    print(symbolic_symmetric_tensor)
    print("\n")