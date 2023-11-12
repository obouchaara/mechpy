import sympy as sp


def sym_matrix_symbol(n):
    M = sp.MatrixSymbol("M", n, n)
    elements = [[0 for _ in range(n)] for _ in range(n)]
        
    for i in range(n):
        for j in range(n):
            if i <= j:
                elements[i][j] = elements[j][i] = M[i, j]
    
    return sp.Matrix(elements)
