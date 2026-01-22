from mpi4py import MPI
import numpy as np

from communicator import nearby_rank_communicator, remainder_communicator
from constants import MPI_DTYPE, MATRIX_DTYPE
from debug import parallel_print, print_full_matrices, print_local_matrices, print_local_matrices_on_debug_rank, print_ranks, rank_print
from distribution import A14_distribution, A9_distribution, C11_get_local_indices, block_cyclic_distribution, col_major_distribution, col_major_distribution_get_local_indices, get_subtile, get_subtile_shape, pure_column_distribution, pure_column_distribution_get_local_indices, pure_row_distribution, pure_row_distribution_get_local_indices, row_major_distribution, row_major_distribution_get_local_indices, set_subtile, alternating_column_distribution, alternating_row_distribution
from util import DoubleBuffer, assemble_matrix_from_tiles, generate_matrices, generate_matrix, matrices_equal, mpi_setup, call_algorithm, create_algorithm_output, send, receive

"""
How to read the functions

We first set the numpy seed to ensure all processors generate identical matrices before they get their distributed portion
Each algorithm begins with a check to ensure the matrix dimensions are able to be properly divided by the processor configuration
We then generate 3 identical matrices across all the processors and each processor will take a small chunk to simulate the matrix being sent out
Each algorithm communicates in two different ways so we select the communicator we need from the various options in communicator.py
Each processor then takes its local part of the matrix using various different distributions that can be found in distribution.py
Once that is done the setup is complete and we start a timer and call the actual algorithm

Each algorithm is comprised of two main loops that go through the two different ways to communicate the matrix
comm1 is defined to be the communicator that fully sends its original tile once comm2 has done a full loop of communication
comm2 will communicate after each local matrix multiplication in the inner loop of the muliplication
Each communicator will send data to the previous rank and receive data from the following rank
We additionally have two index variables called (matrix)_index and (matrix)_index where (matrix) is A, B, or C
These represent the index into the Matrix tile we are currently multiplying with

After the algorithm ends we compute the expected matrix and gather the matrix that was made from the algorithm to compare
We finally return the elapsed time, whether the matrix was multiplied correctly, the A, B, and C matrices and the expected and actual matrix

"""


def AG_A_COL_AG_A_ROW(m, k, n, px, py):
    # 1
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % py == 0
    assert n % size == 0

    A_comm1 = remainder_communicator(comm, py, rank)
    A_comm2 = nearby_rank_communicator(comm, py, rank)

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    A_local = row_major_distribution(A, px, py, rank)
    B_local = pure_column_distribution(B, size, rank)
    C_local = pure_column_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        B_index = comm2_rank
        C_index = comm1_rank
        outer_loop_iterations = px
        buffer = DoubleBuffer(A, make_contiguous=False) # only one buffer needed since col A and row A
        inner_loop_iterations = py

        for i in range(outer_loop_iterations):

            for j in range(inner_loop_iterations):
                
                A_curr = buffer.get_current_tile()
                B_curr = get_subtile(B, py, 1, B_index, 0)

                if j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, A_curr, buffer.get_receive_buffer())
                elif i != outer_loop_iterations - 1:
                    outer_send_request, outer_receive_request = send(comm1, buffer.get_current_tile(), buffer.get_receive_buffer()) 

                C_curr = get_subtile(C, px, 1, C_index, 0)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, px, 1, C_index, 0)     

                if j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: buffer.swap())
                elif i != outer_loop_iterations - 1:
                    receive([outer_send_request, outer_receive_request], lambda: buffer.swap())

                # WHAT IS THIS IF STATEMENT
                if j != inner_loop_iterations - 1:
                    B_index = (B_index + 1) % py

            C_index = (C_index + 1) % px

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, A_comm1, A_comm2, px, py)

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_COL_AG_B_COL(m, k, n, px, py):
    # 2
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % py == 0
    assert n % size == 0

    A_comm = remainder_communicator(comm, px, rank)
    B_comm = nearby_rank_communicator(comm, px, rank)

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C    

    A_local = col_major_distribution(A, px, py, rank)
    B_local = pure_column_distribution(B, size, rank)
    C_local = col_major_distribution(C, px, py, rank)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        C_index = comm2_rank
        B_index = comm1_rank
        outer_loop_iterations = py
        outer_buffer = DoubleBuffer(A, make_contiguous=False)
        inner_loop_iterations = px
        inner_buffer = np.empty(shape=get_subtile_shape(B, py, 1), dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):

            if i != outer_loop_iterations - 1:
                outer_send_request, outer_receive_request = send(comm1, outer_buffer.get_current_tile(), outer_buffer.get_receive_buffer())


            for j in range(inner_loop_iterations):
                
                A_curr = outer_buffer.get_current_tile()
                B_curr = get_subtile(B, py, 1, B_index, 0)

                if j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, B_curr, inner_buffer)

                C_curr = get_subtile(C, 1, px, 0, C_index)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, 1, px, 0, C_index)

                if j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: set_subtile(B, inner_buffer, py, 1, B_index, 0))

                C_index = (C_index + 1) % px

            if i != outer_loop_iterations - 1:
                receive([outer_send_request, outer_receive_request], lambda: outer_buffer.swap())

            B_index = (B_index + 1) % py

        return C

    
    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, A_comm, B_comm, px, py)

    actual_tiles = comm.allgather((C_local, col_major_distribution_get_local_indices(px, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_COL_AG_B_ROW(m, k, n, px, py):
    # 3
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % size == 0
    assert n % py == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    A_comm = nearby_rank_communicator(comm, py, rank)
    B_comm = remainder_communicator(comm, py, rank)

    # maybe change the comm ranks to just be the rank somehow
    A_local = block_cyclic_distribution(A, px, py, B_comm.Get_rank(), A_comm.Get_rank())
    B_local = row_major_distribution(B, px, py, rank)
    C_local = row_major_distribution(C, px, py, rank)

    def algorithm(A, B, C, comm1, comm2, px, py):
        # we always define comm1 to be the outer communicator
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        B_index = comm2_rank
        A_index = comm1_rank
        outer_loop_iterations = px
        outer_buffer = np.empty(shape=get_subtile_shape(B, py, 1), dtype=MATRIX_DTYPE)
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(A, 1, px), dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, 1, px, 0, A_index)

                if j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, np.ascontiguousarray(A_curr, dtype=MATRIX_DTYPE), inner_buffer)

                B_curr = get_subtile(B, py, 1, B_index, 0)

                if i != outer_loop_iterations - 1:
                    outer_send_request, outer_receive_request = send(comm1, B_curr, outer_buffer)

                C = np.matmul(A_curr, B_curr) + C

                if j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: set_subtile(A, inner_buffer, 1, px, 0, A_index))

                if i != outer_loop_iterations - 1:
                    receive([outer_send_request, outer_receive_request], lambda: set_subtile(B, outer_buffer, py, 1, B_index, 0))

                B_index = (B_index + 1) % py

            A_index = (A_index + 1) % px

        return C


    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, B_comm, A_comm, px, py)

    actual_tiles = comm.allgather((C_local, row_major_distribution_get_local_indices(py, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_COL_RS_C_COL(m, k, n, px, py):
    # 4
    comm, size, rank = mpi_setup()

    assert m >= 1
    assert k % size == 0
    assert n % size == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    A_comm = remainder_communicator(comm, py, rank)
    C_comm = nearby_rank_communicator(comm, py, rank)

    A_local = alternating_column_distribution(A, px, py, rank)
    B_local = col_major_distribution(B, py, px, rank) # idk why px py are flipped but it works
    C_local = pure_column_distribution(C, size, rank) 

    # print_local_matrices_on_debug_rank(A, B, C)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        B_row_index = comm2_rank
        B_col_index = (comm1_rank + 1) % py # this plus one here and in its update is super trippy and not well communicated in the thesis
        outer_loop_iterations = py
        outer_buffer = DoubleBuffer(np.zeros(shape=C.shape, dtype=MATRIX_DTYPE), make_contiguous=False)
        inner_loop_iterations = px
        inner_buffer = DoubleBuffer(A, make_contiguous=True)

        for i in range(outer_loop_iterations):
            
            C_temp = np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)

            for j in range(inner_loop_iterations):

                A_curr = inner_buffer.get_current_tile()

                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1: # NOTICE outer loops need to reset for next big loop
                    inner_send_request, inner_receive_request = send(comm2, inner_buffer.get_current_tile(), inner_buffer.get_receive_buffer())

                B_curr = get_subtile(B, px, py, B_row_index, B_col_index)
                C_temp = np.matmul(A_curr, B_curr) + C_temp

                # print_local_matrices_on_debug_rank(A_curr, B_curr, C_curr, debug_rank=0)

                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: inner_buffer.swap())
                
                B_row_index = (B_row_index + 1) % px

            if i == 0:
                C_curr = outer_buffer.get_current_tile()
            else:
                C_curr = receive([outer_receive_request, outer_send_request], lambda: (outer_buffer.swap(), outer_buffer.get_current_tile())[1])

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C += C_curr
            else:
                outer_send_request, outer_receive_request = send(comm1, C_curr, outer_buffer.get_receive_buffer())

            B_col_index = (B_col_index + 1) % py

        return C
            
    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, A_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_COL_RS_C_ROW(m, k, n, px, py):
    # 5
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % size == 0
    assert n % py == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    A_comm = remainder_communicator(comm, px, rank)
    C_comm = nearby_rank_communicator(comm, px, rank)

    A_local = alternating_column_distribution(A, py, px, rank)
    B_local = col_major_distribution(B, px, py, rank)
    C_local = col_major_distribution(C, px, py, rank) 

    # print_local_matrices_on_debug_rank(A, B, C, debug_rank=3)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=3)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        A_index = (comm1_rank + 1) % px
        B_index = comm2_rank
        outer_loop_iterations = px
        outer_buffer = DoubleBuffer(np.zeros(shape=C.shape, dtype=MATRIX_DTYPE), make_contiguous=True)
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(A, px, 1), dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):

            A_curr = get_subtile(A, px, 1, A_index, 0)

            C_temp = np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)

            for j in range(inner_loop_iterations):
                
                if j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, np.ascontiguousarray(A_curr, dtype=MATRIX_DTYPE), inner_buffer)

                B_curr = get_subtile(B, py, 1, B_index, 0)

                C_temp = np.matmul(A_curr, B_curr) + C_temp

                if j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: np.copyto(A_curr, inner_buffer))

                B_index = (B_index + 1) % py # I think there is a typo in the thesis, #5 shows B_index - 1 here

            if i == 0:
                C_curr = outer_buffer.get_current_tile()
            else:
                C_curr = receive([outer_receive_request, outer_send_request], lambda: (outer_buffer.swap(), outer_buffer.get_current_tile())[1])

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C += C_curr
            else:
                # The arrows in the paper show this as a plus I think (the comm direction)
                outer_send_request, outer_receive_request = send(comm1, C_curr, outer_buffer.get_receive_buffer())
            
            A_index = (A_index + 1) % px

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, A_comm, px, py) 

    actual_tiles = comm.allgather((C_local, col_major_distribution_get_local_indices(px, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_ROW_AG_B_COL(m, k, n, px, py):
    # 6
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k >= 1
    assert n % size == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C    

    A_comm = remainder_communicator(comm, py, rank)
    B_comm = nearby_rank_communicator(comm, py, rank)

    A_local = alternating_row_distribution(A, px, py, rank)
    B_local = pure_column_distribution(B, size, rank)
    C_local = col_major_distribution(C, py, px, rank)

    # print_local_matrices_on_debug_rank(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=0)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        C_row_index = comm1_rank
        C_col_index = comm2_rank
        outer_loop_iterations = px
        outer_buffer = DoubleBuffer(A, make_contiguous=False)
        inner_loop_iterations = py
        inner_buffer = DoubleBuffer(B, make_contiguous=False)

        for i in range(outer_loop_iterations):

            A_curr = outer_buffer.get_current_tile()

            if i != outer_loop_iterations - 1:
                outer_send_request, outer_receive_request = send(comm1, outer_buffer.get_current_tile(), outer_buffer.get_receive_buffer())

            for j in range(inner_loop_iterations):

                B_curr = inner_buffer.get_current_tile()

                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, inner_buffer.get_current_tile(), inner_buffer.get_receive_buffer())

                C_curr = get_subtile(C, px, py, C_row_index, C_col_index)
                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                # print_local_matrices_on_debug_rank(A_curr, B_curr, C_tmp, debug_rank=0)
                set_subtile(C, C_tmp, px, py, C_row_index, C_col_index)

                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: inner_buffer.swap())

                C_col_index = (C_col_index + 1) % py

            if i != outer_loop_iterations - 1:
                receive([outer_send_request, outer_receive_request], lambda: outer_buffer.swap())

            C_row_index = (C_row_index + 1) % px

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, A_comm, B_comm, px, py)

    actual_tiles = comm.allgather((C_local, col_major_distribution_get_local_indices(py, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_ROW_AG_B_ROW(m, k, n, px, py):
    # 7
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k % px == 0
    assert n % py == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C    

    A_comm = nearby_rank_communicator(comm, py, rank)
    B_comm = remainder_communicator(comm, py, rank)

    A_local = pure_row_distribution(A, size, rank)
    B_local = row_major_distribution(B, px, py, rank)
    C_local = row_major_distribution(C, px, py, rank)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        C_index = comm2_rank
        A_index = comm1_rank
        outer_loop_iterations = px
        outer_buffer = DoubleBuffer(B, make_contiguous=False)
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(A, 1, px), dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):
            
            B_curr = outer_buffer.get_current_tile()

            if i != outer_loop_iterations - 1:
                outer_send_request, outer_receive_request = send(comm1, outer_buffer.get_current_tile(), outer_buffer.get_receive_buffer())
  

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, 1, px, 0, A_index)

                if j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, np.ascontiguousarray(A_curr, dtype=MATRIX_DTYPE), inner_buffer)
 

                C_curr = get_subtile(C, py, 1, C_index, 0)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, py, 1, C_index, 0)

                if j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: set_subtile(A, inner_buffer, 1, px, 0, A_index))
  

                C_index = (C_index + 1) % py       

            if i != outer_loop_iterations - 1:
                receive([outer_send_request, outer_receive_request], lambda: outer_buffer.swap())
       
            A_index = (A_index + 1) % px
        
        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, B_comm, A_comm, px, py)

    actual_tiles = comm.allgather((C_local, row_major_distribution_get_local_indices(py, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_ROW_RS_C_COL(m, k, n, px, py):
    # 8
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % py == 0
    assert n % size == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    A_comm = remainder_communicator(comm, py, rank)
    C_comm = nearby_rank_communicator(comm, py, rank)

    A_local = row_major_distribution(A, px, py, rank)
    B_local = col_major_distribution(B, py, px, rank) # flipped!
    C_local = pure_column_distribution(C, size, rank) 

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        B_index = (comm2_rank + 1) % py
        C_index = comm1_rank
        outer_loop_iterations = px
        outer_buffer = DoubleBuffer(A, make_contiguous=False)
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(C, px, 1), dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):

            A_curr = outer_buffer.get_current_tile()

            if i != outer_loop_iterations - 1:
                outer_send_request, outer_receive_request = send(comm1, outer_buffer.get_current_tile(), outer_buffer.get_receive_buffer())

            for j in range(inner_loop_iterations):

                B_curr = get_subtile(B, 1, py, 0, B_index)

                C_temp = np.matmul(A_curr, B_curr)

                if j == 0:
                    C_curr = np.zeros(shape=get_subtile_shape(C, px, 1), dtype=MATRIX_DTYPE)            
                else:
                    C_curr = receive([inner_receive_request, inner_send_request], lambda: inner_buffer)

                C_curr = C_curr + C_temp

                if j == inner_loop_iterations - 1:
                    C_curr = C_curr + get_subtile(C, px, 1, C_index, 0)   
                    set_subtile(C, C_curr, px, 1, C_index, 0)
                else:
                    inner_send_request, inner_receive_request = send(comm2, C_curr, inner_buffer)

                B_index = (B_index + 1) % py 

            if i != outer_loop_iterations - 1:
                receive([outer_send_request, outer_receive_request], lambda: outer_buffer.swap())
 
            C_index = (C_index + 1) % px
        
        return C


    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, A_comm, C_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)
    

def AG_A_ROW_RS_C_ROW(m, k, n, px, py):
    # 9
    # there is a typo in the drawing, bottom row, p0 for p1 should be p1 for p1
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k % px == 0
    assert n % py == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    A_comm = remainder_communicator(comm, px, rank)
    C_comm = nearby_rank_communicator(comm, px, rank)

    A_local = A9_distribution(A, px, py, rank)
    B_local = col_major_distribution(B, px, py, rank)
    C_local = col_major_distribution(C, px, py, rank) 

    # print_local_matrices_on_debug_rank(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=5)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        A_index = (comm1_rank + 1) % px
        C_index = comm2_rank
        outer_loop_iterations = px
        outer_buffer = DoubleBuffer(np.zeros(C.shape), make_contiguous=True)
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(A, px, 1), dtype=MATRIX_DTYPE)

        B_curr = B

        for i in range(outer_loop_iterations):
            
            C_temp = np.zeros(C.shape, dtype=MATRIX_DTYPE)

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, px, 1, A_index, 0)

                if j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, A_curr, inner_buffer)

                C_temp_temp = get_subtile(C_temp, py, 1, C_index, 0)
                C_temp_temp = np.matmul(A_curr, B_curr) + C_temp_temp
                set_subtile(C_temp, C_temp_temp, py, 1, C_index, 0)
                # print_local_matrices_on_debug_rank(A_curr, C_curr_curr, C_curr, debug_rank=1)

                if j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: set_subtile(A, inner_buffer, px, 1, A_index, 0))

                C_index = (C_index + 1) % py

            if i == 0:
                # C_curr is the whole C block
                C_curr = outer_buffer.get_current_tile()
            else:
                C_curr = receive([outer_send_request, outer_receive_request], lambda: (outer_buffer.swap(), outer_buffer.get_current_tile())[1])

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C = C + C_curr
            else:
                outer_send_request, outer_receive_request = send(comm1, C_curr, outer_buffer.get_receive_buffer())

            A_index = (A_index + 1) % px

        return C
    
    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, A_comm, px, py)

    actual_tiles = comm.allgather((C_local, col_major_distribution_get_local_indices(px, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_B_COL_AG_B_ROW(m, k, n, px, py):
    # 10
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k % px == 0
    assert n % py == 0

    B_comm1 = remainder_communicator(comm, px, rank)
    B_comm2 = nearby_rank_communicator(comm, px, rank)

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C  

    A_local = pure_row_distribution(A, size, rank)
    B_local = col_major_distribution(B, px, py, rank)
    C_local = pure_row_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        A_index = comm2_rank
        C_index = comm1_rank
        outer_loop_iterations = py
        buffer = DoubleBuffer(B, make_contiguous=False)
        inner_loop_iterations = px

        for i in range(outer_loop_iterations):

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, 1, px, 0, A_index)
                B_curr = buffer.get_current_tile()

                if j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, buffer.get_current_tile(), buffer.get_receive_buffer())
                elif i != outer_loop_iterations - 1:
                    outer_send_request, outer_receive_request = send(comm1, buffer.get_current_tile(), buffer.get_receive_buffer())

                C_curr = get_subtile(C, 1, py, 0, C_index)

                # print_local_matrices_on_debug_rank(A_curr, B_curr, C_curr)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, 1, py, 0, C_index)

                if j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: buffer.swap())
                elif i != outer_loop_iterations - 1:
                    receive([outer_send_request, outer_receive_request], lambda: buffer.swap())

                if j != inner_loop_iterations - 1:
                    A_index = (A_index + 1) % px

            C_index = (C_index + 1) % py
        
        return C

    # print_full_matrices(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local)

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, B_comm1, B_comm2, px, py)

    actual_tiles = comm.allgather((C_local, pure_row_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_B_COL_RS_C_COL(m, k, n, px, py):
    # 11
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % py == 0
    assert n % size == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    B_comm = remainder_communicator(comm, py, rank)
    C_comm = nearby_rank_communicator(comm, py, rank)

    A_local = row_major_distribution(A, px, py, rank)
    B_local = col_major_distribution(B, py, px, rank)
    C_local = block_cyclic_distribution(C, px, py, B_comm.Get_rank(), C_comm.Get_rank()) 

    # print_local_matrices_on_debug_rank(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=5)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()

        B_index = (comm1_rank + 1) % py
        C_index = comm2_rank
        outer_loop_iterations = py
        outer_buffer = DoubleBuffer(np.zeros(shape=C.shape), make_contiguous=True)
        inner_loop_iterations = px
        # inner_buffer = DoubleBuffer(B, make_contiguous=False)
        inner_buffer = np.empty(shape=get_subtile_shape(B, 1, py), dtype=MATRIX_DTYPE)

        A_curr = A

        for i in range(outer_loop_iterations):
            
            C_temp = np.zeros(C.shape, dtype=MATRIX_DTYPE)

            for j in range(inner_loop_iterations):
                
                B_curr = get_subtile(B, 1, py, 0, B_index)

                if i != outer_loop_iterations - 1 or j != inner_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, np.ascontiguousarray(B_curr, dtype=MATRIX_DTYPE), inner_buffer)
                
                # B_curr_curr = get_subtile(B_curr, 1, py, 0, B_index)
                C_temp_temp = get_subtile(C_temp, 1, px, 0, C_index)

                C_temp_temp = np.matmul(A_curr, B_curr) + C_temp_temp
                # C_temp_temp = np.matmul(A_curr, B_curr_curr) + C_temp_temp
                set_subtile(C_temp, C_temp_temp, 1, px, 0, C_index)

                # print_local_matrices_on_debug_rank(A_curr, B_curr_curr, C_curr, debug_rank=0)

                if i != outer_loop_iterations - 1 or j != inner_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: set_subtile(B, inner_buffer, 1, py, 0, B_index))

                C_index = (C_index + 1) % px

            if i == 0:
                C_curr = outer_buffer.get_current_tile()
            else:
                C_curr = receive([outer_send_request, outer_receive_request], lambda: (outer_buffer.swap(), outer_buffer.get_current_tile())[1])

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C = C + C_curr
            else:
                outer_send_rank = (comm1_rank - 1) % comm1.Get_size()
                outer_receive_rank = (comm1_rank + 1) % comm1.Get_size()
                outer_send_request = comm1.Isend(
                    buf=(C_curr, MPI_DTYPE), 
                    dest=outer_send_rank
                )
                outer_receive_request = comm1.Irecv(
                    buf=(outer_buffer.get_receive_buffer(), MPI_DTYPE), 
                    source=outer_receive_rank
                )       

            B_index = (B_index + 1) % py

        return C
    
    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, B_comm, px, py) 

    actual_tiles = comm.allgather(C11_get_local_indices(C_local, px, py, B_comm.Get_rank(), C_comm.Get_rank()))
    flattened = [item for sublist in actual_tiles for item in sublist]
    actual = assemble_matrix_from_tiles(flattened)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_B_COL_RS_C_ROW(m, k, n, px, py):
    # 12
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k % px == 0
    assert n % py == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    B_comm = remainder_communicator(comm, px, rank)
    C_comm = nearby_rank_communicator(comm, px, rank)

    A_local = row_major_distribution(A, py, px, rank)
    B_local = col_major_distribution(B, px, py, rank)
    C_local = pure_row_distribution(C, size, rank)

    # print_local_matrices_on_debug_rank(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=3)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        A_index = (comm2_rank + 1) % px
        C_index = comm1_rank
        outer_loop_iterations = py
        outer_buffer = DoubleBuffer(B, make_contiguous=False)
        inner_loop_iterations = px
        inner_buffer = np.empty(shape=get_subtile_shape(C, 1, py), dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):
            B_curr = outer_buffer.get_current_tile()

            if i != outer_loop_iterations - 1:
                outer_send_request, outer_receive_request = send(comm1, outer_buffer.get_current_tile(), outer_buffer.get_receive_buffer())

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, px, 1, A_index, 0)

                C_temp = np.matmul(A_curr, B_curr)

                if j == 0:
                    C_curr = np.zeros(shape=get_subtile_shape(C, 1, py), dtype=MATRIX_DTYPE)          
                else:
                    C_curr = receive([inner_receive_request, inner_send_request], lambda: inner_buffer)

                C_curr = C_curr + C_temp

                if j == inner_loop_iterations - 1:
                    C_curr = C_curr + get_subtile(C, 1, py, 0, C_index)
                    set_subtile(C, C_curr, 1, py, 0, C_index)
                else:
                    inner_send_request, inner_receive_request = send(comm2, C_curr, inner_buffer)

                A_index = (A_index + 1) % px

            if i != outer_loop_iterations - 1:
                receive([outer_send_request, outer_receive_request], lambda: outer_buffer.swap())

            C_index = (C_index + 1) % py     

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, B_comm, C_comm, px, py) 

    actual_tiles = comm.allgather((C_local, pure_row_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_B_ROW_RS_C_COL(m, k, n, px, py):
    # 13
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % size == 0
    assert n % py == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    B_comm = remainder_communicator(comm, py, rank)
    C_comm = nearby_rank_communicator(comm, py, rank)

    A_local = block_cyclic_distribution(A, px, py, B_comm.Get_rank(), C_comm.Get_rank())
    B_local = pure_row_distribution(B, size, rank)
    C_local = row_major_distribution(C, px, py, rank)

    # print_local_matrices_on_debug_rank(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=0)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        A_index = comm2_rank
        B_index = (comm1_rank + 1) % py
        outer_loop_iterations = py
        outer_buffer = DoubleBuffer(np.zeros(shape=C.shape, dtype=MATRIX_DTYPE), make_contiguous=False)
        inner_loop_iterations = px
        inner_buffer = np.empty(shape=get_subtile_shape(B, 1, py), dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):
            B_curr = get_subtile(B, 1, py, 0, B_index)
            C_temp = np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)

            for j in range(inner_loop_iterations):

                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1:
                    inner_send_request, inner_receive_request = send(comm2, np.ascontiguousarray(B_curr, dtype=MATRIX_DTYPE), inner_buffer)

                A_curr = get_subtile(A, 1, px, 0, A_index)
                C_temp = np.matmul(A_curr, B_curr) + C_temp
                # print_local_matrices_on_debug_rank(A_curr, B_curr, C_temp, debug_rank=0)

                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: np.copyto(B_curr, inner_buffer))

                A_index = (A_index + 1) % px

            if i == 0:
                C_curr = outer_buffer.get_current_tile()
            else:
                C_curr = receive([outer_receive_request, outer_send_request], lambda: (outer_buffer.swap(), outer_buffer.get_current_tile())[1])

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C = C + C_curr
            else:
                outer_send_request, outer_receive_request = send(comm1, C_curr, outer_buffer.get_receive_buffer())

            B_index = (B_index + 1) % py

        return C
    
    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, B_comm, px, py)

    actual_tiles = comm.allgather((C_local, row_major_distribution_get_local_indices(py, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_B_ROW_RS_C_ROW(m, k, n, px, py):
    # 14
    # need to add arrows for B
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k % size == 0
    assert n >= 1

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    B_comm = remainder_communicator(comm, px, rank)
    C_comm = nearby_rank_communicator(comm, px, rank)
    
    # A_local = A14_distribution(A, py, px, rank)
    A_local = block_cyclic_distribution(A, py, px, B_comm.Get_rank(), C_comm.Get_rank())
    B_local = pure_row_distribution(B, size, rank)
    C_local = pure_row_distribution(C, size, rank)

    # print_local_matrices_on_debug_rank(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=1)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        A_col_index = comm2.Get_rank()
        A_row_index = (comm1.Get_rank() + 1) % px 
        outer_loop_iterations = px
        outer_buffer = DoubleBuffer(np.zeros(shape=C.shape, dtype=MATRIX_DTYPE), make_contiguous=False)
        inner_loop_iterations = py
        inner_buffer = DoubleBuffer(B, make_contiguous=False)

        for i in range(outer_loop_iterations):

            C_temp = np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)

            for j in range(inner_loop_iterations):
                
                B_curr = inner_buffer.get_current_tile()

                # the diagram in the thesis is missing all of the arrows for the last 4 pictures
                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1: # must loop B back around to start
                    inner_send_request, inner_receive_request = send(comm2, inner_buffer.get_current_tile(), inner_buffer.get_receive_buffer())

                A_curr = get_subtile(A, px, py, A_row_index, A_col_index)
                C_temp = np.matmul(A_curr, B_curr) + C_temp
                # rank_print(f"A_row: {A_row_index}, A_col: {A_col_index}", print_rank=1)
                # print_local_matrices_on_debug_rank(A_curr, B_curr, C_curr, debug_rank=1)

                if j != inner_loop_iterations - 1 or i != outer_loop_iterations - 1:
                    receive([inner_send_request, inner_receive_request], lambda: inner_buffer.swap())

                A_col_index = (A_col_index + 1) % py

            if i == 0:
                C_curr = outer_buffer.get_current_tile()
            else:
                C_curr = receive([outer_receive_request, outer_send_request], lambda: (outer_buffer.swap(), outer_buffer.get_current_tile())[1])

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C = C + C_curr
            else:
                outer_send_request, outer_receive_request = send(comm1, C_curr, outer_buffer.get_receive_buffer())

            A_row_index = (A_row_index + 1) % px

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, B_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_row_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def RS_C_COL_RS_C_ROW(m, k, n, px, py):
    # 15
    comm, size, rank = mpi_setup()

    assert m % px == 0
    assert k % size == 0
    assert n % py == 0

    C_comm1 = nearby_rank_communicator(comm, px, rank)
    C_comm2 = remainder_communicator(comm, px, rank)

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C    

    A_local = pure_column_distribution(A, size, rank)
    B_local = pure_row_distribution(B, size, rank)
    C_local = col_major_distribution(C, px, py, rank)

    # print_local_matrices_on_debug_rank(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local, debug_rank=5)

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        A_index = (comm1_rank + 1) % px
        # B_index = (comm2_rank - 1) % py
        B_index = (comm2_rank + px) % py
        outer_loop_iterations = px
        buffer = DoubleBuffer(np.zeros(shape=C.shape), make_contiguous=True) # only one buffer needed since just moving C
        inner_loop_iterations = py

        for i in range(outer_loop_iterations):
            
            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, px, 1, A_index, 0)
                B_curr = get_subtile(B, 1, py, 0, B_index)

                # print_local_matrices_on_debug_rank(A_curr, B_curr, C_curr, debug_rank=0)
                C_temp = np.matmul(A_curr, B_curr)
                
                if i == 0 and j == 0:
                    C_curr = buffer.get_current_tile()
                else:
                    C_curr = receive([send_request, receive_request], lambda: (buffer.swap(), buffer.get_current_tile())[1])

                C_curr = C_curr + C_temp

                if i == outer_loop_iterations - 1 and j == inner_loop_iterations - 1:
                    C = C + C_curr
                elif j == inner_loop_iterations - 1:
                    send_request, receive_request = send(comm1, C_curr, buffer.get_receive_buffer())
                else:
                    send_request, receive_request = send(comm2, C_curr, buffer.get_receive_buffer())

                if j != inner_loop_iterations - 1:
                    B_index = (B_index + 1) % py

            A_index = (A_index + 1) % px

        return C
    
    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm1, C_comm2, px, py)

    actual_tiles = comm.allgather((C_local, col_major_distribution_get_local_indices(px, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


# 1d algorithms
def AG_A_COL(m, k, n, px, py):
    comm, size, rank = mpi_setup()

    assert m >= 1
    assert k % size == 0
    assert n % size == 0

    A, B, C = generate_matrices(m, k, n)
    expected = np.matmul(A, B) + C

    A_comm = comm

    A_local = pure_column_distribution(A, size, rank)
    B_local = pure_column_distribution(B, size, rank)
    C_local = pure_column_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, px, py):
        comm1_rank = comm1.Get_rank()
        comm1_size = comm1.Get_size()
        B_index = comm1_rank
        outer_loop_iterations = comm1_size
        buffer = DoubleBuffer(A, make_contiguous=False)

        for i in range(outer_loop_iterations):

            if i != outer_loop_iterations - 1:
                send_request, receive_request = send(comm1, buffer.get_current_tile(), buffer.get_receive_buffer())

            A_curr = buffer.get_current_tile()
            B_curr = get_subtile(B, size, 1, B_index, 0)

            C = C + np.matmul(A_curr, B_curr)

            if i != outer_loop_iterations - 1:
                receive([send_request, receive_request], lambda: buffer.swap())

            B_index = (B_index + 1) % comm1_size

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, A_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_A_ROW(m, k, n, px, py):
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k >= 1
    assert n % size == 0

    A, B, C = generate_matrices(m, k, n)
    expected = np.matmul(A, B) + C

    A_comm = comm

    A_local = pure_row_distribution(A, size, rank)
    B_local = pure_column_distribution(B, size, rank)
    C_local = pure_column_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, px, py):
        comm1_rank = comm1.Get_rank()
        comm1_size = comm1.Get_size()
        C_index = comm1_rank
        outer_loop_iterations = comm1_size
        buffer = DoubleBuffer(A, make_contiguous=False)

        for i in range(outer_loop_iterations):

            if i != outer_loop_iterations - 1:
                send_request, receive_request = send(comm1, buffer.get_current_tile(), buffer.get_receive_buffer())

            A_curr = buffer.get_current_tile()
            
            local_result = np.matmul(A_curr, B)
            C_curr = get_subtile(C, size, 1, C_index, 0)
            C_tmp = local_result + C_curr
            set_subtile(C, C_tmp, size, 1, C_index, 0)

            if i != outer_loop_iterations - 1:
                receive([send_request, receive_request], lambda: buffer.swap())

            C_index = (C_index + 1) % comm1_size

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, A_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_B_COL(m, k, n, px, py):
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k >= 1
    assert n % size == 0

    A, B, C = generate_matrices(m, k, n)
    expected = np.matmul(A, B) + C

    B_comm = comm

    A_local = pure_row_distribution(A, size, rank)
    B_local = pure_column_distribution(B, size, rank)
    C_local = pure_row_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, px, py):
        comm1_rank = comm1.Get_rank()
        comm1_size = comm1.Get_size()
        C_index = comm1_rank
        outer_loop_iterations = comm1_size
        buffer = DoubleBuffer(B, make_contiguous=False)

        for i in range(outer_loop_iterations):

            if i != outer_loop_iterations - 1:
                send_rank = (comm1_rank - 1) % comm1_size
                receive_rank = (comm1_rank + 1) % comm1_size
                send_request = comm1.Isend(
                    buf=(buffer.get_current_tile(), MPI_DTYPE), dest=send_rank
                )
                receive_request = comm1.Irecv(
                    buf=(buffer.get_receive_buffer(), MPI_DTYPE), source=receive_rank
                )

            B_curr = buffer.get_current_tile()
            
            local_result = np.matmul(A, B_curr)
            C_curr = get_subtile(C, 1, size, 0, C_index)
            C_tmp = local_result + C_curr
            set_subtile(C, C_tmp, 1, size, 0, C_index)

            if i != outer_loop_iterations - 1:
                MPI.Request.Waitall([send_request, receive_request])
                buffer.swap()

            C_index = (C_index + 1) % comm1_size

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, B_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_row_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def AG_B_ROW(m, k, n, px, py):
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k % size == 0
    assert n >= 1

    A, B, C = generate_matrices(m, k, n)
    expected = np.matmul(A, B) + C

    B_comm = comm

    A_local = pure_row_distribution(A, size, rank)
    B_local = pure_row_distribution(B, size, rank)
    C_local = pure_row_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, px, py):
        comm1_rank = comm1.Get_rank()
        comm1_size = comm1.Get_size()
        A_index = comm1_rank
        outer_loop_iterations = comm1_size
        buffer = DoubleBuffer(B, make_contiguous=False)

        for i in range(outer_loop_iterations):

            if i != outer_loop_iterations - 1:
                send_request, receive_request = send(comm1, buffer.get_current_tile(), buffer.get_receive_buffer())

            B_curr = buffer.get_current_tile()
            A_curr = get_subtile(A, 1, size, 0, A_index)

            C = C + np.matmul(A_curr, B_curr)

            if i != outer_loop_iterations - 1:
                receive([send_request, receive_request], lambda: buffer.swap())

            A_index = (A_index + 1) % comm1_size

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, B_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_row_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def RS_C_COL(m, k, n, px, py):
    comm, size, rank = mpi_setup()

    assert m >= 1
    assert k % size == 0
    assert n % size == 0

    A, B, C = generate_matrices(m, k, n)
    expected = np.matmul(A, B) + C

    C_comm = comm

    A_local = pure_column_distribution(A, size, rank)
    B_local = pure_row_distribution(B, size, rank)
    C_local = pure_column_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, px, py):
        comm1_rank = comm1.Get_rank()
        comm1_size = comm1.Get_size()
        B_index = (comm1_rank + 1) % comm1_size
        outer_loop_iterations = comm1_size
        buffer = np.empty(shape=C.shape, dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):

            B_curr = get_subtile(B, 1, size, 0, B_index)
            
            C_temp = np.matmul(A, B_curr)

            if i == 0:
                C_curr = np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
            else:
                C_curr = receive([receive_request, send_request], lambda: buffer)

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C = C + C_curr
            else:
                send_request, receive_request = send(comm1, C_curr, buffer)

            B_index = (B_index + 1) % comm1_size

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)


def RS_C_ROW(m, k, n, px, py):
    comm, size, rank = mpi_setup()

    assert m % size == 0
    assert k % size == 0
    assert n >= 1

    A, B, C = generate_matrices(m, k, n)
    expected = np.matmul(A, B) + C

    C_comm = comm

    A_local = pure_column_distribution(A, size, rank)
    B_local = pure_row_distribution(B, size, rank)
    C_local = pure_row_distribution(C, size, rank)

    def algorithm(A, B, C, comm1, px, py):
        comm1_rank = comm1.Get_rank()
        comm1_size = comm1.Get_size()
        A_index = (comm1_rank + 1) % comm1_size
        outer_loop_iterations = comm1_size
        buffer = np.empty(shape=C.shape, dtype=MATRIX_DTYPE)

        for i in range(outer_loop_iterations):

            A_curr = get_subtile(A, size, 1, A_index, 0)
            
            C_temp = np.matmul(A_curr, B)

            if i == 0:
                C_curr = np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
            else:
                C_curr = receive([receive_request, send_request], lambda: buffer)

            C_curr = C_curr + C_temp

            if i == outer_loop_iterations - 1:
                C = C + C_curr
            else:
                send_request, receive_request = send(comm1, C_curr, buffer)

            A_index = (A_index + 1) % comm1_size

        return C

    C_local, elapsed_time = call_algorithm(algorithm, comm, A_local, B_local, C_local, C_comm, px, py)

    actual_tiles = comm.allgather((C_local, pure_row_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)
