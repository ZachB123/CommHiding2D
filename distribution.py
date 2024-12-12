import numpy as np

def get_subtile(tile, rows, columns, i, j):
    # we split the tile into the rows and columns and can index it like a 2d array
    # if column is 1 then we just use j as an index

    subtile_rows = tile.shape[0] // rows
    subtile_columns = tile.shape[1] // columns

    start_row = i * subtile_rows
    end_row = start_row + subtile_rows

    start_col = j * subtile_columns
    end_col = start_col + subtile_columns

    return tile[start_row:end_row, start_col:end_col]


def get_subtile_shape(tile, rows, columns, i, j):
    subtile_rows = tile.shape[0] // rows
    subtile_columns = tile.shape[1] // columns

    return (subtile_rows, subtile_columns)


def set_subtile(tile, subtile, rows, columns, i, j):
    subtile_rows = tile.shape[0] // rows
    subtile_columns = tile.shape[1] // columns

    start_row = i * subtile_rows
    end_row = start_row + subtile_rows

    start_col = j * subtile_columns
    end_col = start_col + subtile_columns

    # Set the subtile in the original tile
    tile[start_row:end_row, start_col:end_col] = subtile


def row_major_distribution(matrix, px, py, rank):
    # labels going across a row first
    return get_subtile(matrix, px, py, rank // py, rank % py).copy()

def row_major_distribution_get_local_indices(py, rank):
    return (rank // py, rank % py)

def col_major_distribution(matrix, px, py, rank):
    # label going down a a column first
    return get_subtile(matrix, px, py, rank % px, rank // px).copy()

def col_major_distribution_get_local_indices(px, rank):
    return (rank % px, rank // px)

def pure_column_distribution(matrix, size, rank):
    return get_subtile(matrix, 1, size, 0, rank).copy()

def pure_column_distribution_get_local_indices(rank):
    return (0, rank)

def pure_row_distribution(rank):
    return (rank, 0)

def block_cyclic_distribution(matrix, px, py, row_index, start_col_index):
    # used in 3
    p = px * py
    subtiles = []
    curr_index = start_col_index
    while curr_index < p:
        subtiles.append(get_subtile(matrix, px, p, row_index, curr_index))
        curr_index += py

    return np.hstack(subtiles)