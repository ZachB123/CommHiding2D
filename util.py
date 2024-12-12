import numpy as np

from constants import MATRIX_DTYPE

def generate_matrix(row, col, min, max):
    """
    Generate a matrix of specified shape with random integers.

    Args:
        row (int): Number of rows in the matrix.
        col (int): Number of columns in the matrix.
        min (int): Minimum integer (inclusive).
        max (int): Maximum integer (exclusive).

    Returns:
        np.ndarray: A matrix of shape (row, col) with random integers in [min, max).
    """
    return np.random.randint(min, max, size=(row, col)).astype(MATRIX_DTYPE, copy=False)


def generate_matrices(m, k, n):
    # A, B, C
    A = generate_matrix(m, k, -9, 10,)
    B = generate_matrix(k, n, -9, 10,)
    # C = generate_matrix(m, n, -9, 10,)
    C = np.zeros(shape=(m, n))
    return A, B, C

def matrices_equal(A, B):
    return np.all(np.isclose(A, B))

class DoubleBuffer:
    def __init__(self, initial_value):
        self.first_buffer = initial_value
        self.second_buffer = np.empty(initial_value.shape)
        self.current_buffer = self.first_buffer

    def get_receive_buffer(self):
        # this is wait we can use in the mpi receive request
        # we receive into not the current buffer then swap it
        if self.current_buffer is self.first_buffer:
            return self.second_buffer
        else:
            return self.first_buffer

    def get_current_tile(self):
        return self.current_buffer

    def swap(self):
        if self.current_buffer is self.first_buffer:
            self.current_buffer = self.second_buffer
        else:
            self.current_buffer = self.first_buffer


def assemble_matrix_from_tiles(tiles):
    
    sorted_tiles = sorted(tiles, key=lambda x: (x[1][0], x[1][1]))
    
    matrix_rows = []
    current_row = []
    current_row_idx = 0
    
    for tile, (row_idx, col_idx) in sorted_tiles:
        if row_idx > current_row_idx:
            matrix_rows.append(np.hstack(current_row))
            current_row = [tile]
            current_row_idx = row_idx
        else:
            current_row.append(tile)
    
    if current_row:
        matrix_rows.append(np.hstack(current_row))
    
    return np.vstack(matrix_rows) 