import sympy as sp

def to_strain_voigt_symbolic(matrix: sp.MatrixSymbol):
    return [
        matrix[0, 0],
        matrix[1, 1],
        matrix[2, 2],
        2 * matrix[1, 2],
        2 * matrix[0, 2],
        2 * matrix[0, 1],
    ]