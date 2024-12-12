from mpi4py import MPI
import numpy as np

from communicator import nearby_rank_communicator, remainder_communicator
from constants import MPI_DTYPE
from debug import parallel_print, print_full_matrices, print_local_matrices, print_local_matrices_on_debug_rank, print_ranks, rank_print
from distribution import block_cyclic_distribution, col_major_distribution, col_major_distribution_get_local_indices, get_subtile, get_subtile_shape, pure_column_distribution, pure_column_distribution_get_local_indices, pure_row_distribution, pure_row_distribution_get_local_indices, row_major_distribution, row_major_distribution_get_local_indices, set_subtile
from util import DoubleBuffer, assemble_matrix_from_tiles, generate_matrices, matrices_equal

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
    np.random.seed(42)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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
        buffer = DoubleBuffer(A) # only one buffer needed since col A and row A
        inner_loop_iterations = py

        for i in range(outer_loop_iterations):

            for j in range(inner_loop_iterations):
                
                A_curr = buffer.get_current_tile()
                B_curr = get_subtile(B, py, 1, B_index, 0)

                if j != inner_loop_iterations - 1:
                    inner_send_rank = (comm2_rank - 1) % comm2.Get_size()
                    inner_receive_rank = (comm2_rank + 1) % comm2.Get_size()
                    inner_send_request = comm2.Isend(
                        buf=(np.ascontiguousarray(A_curr), MPI_DTYPE), dest=inner_send_rank
                    )
                    inner_receive_request = comm2.Irecv(
                        buf=(buffer.get_receive_buffer(), MPI_DTYPE), source=inner_receive_rank
                    )
                elif i != outer_loop_iterations - 1:
                    outer_send_rank = (comm1_rank - 1) % comm1.Get_size()
                    outer_receive_rank = (comm1_rank + 1) % comm1.Get_size()
                    outer_send_request = comm1.Isend(
                        buf=(np.ascontiguousarray(buffer.get_current_tile()), MPI_DTYPE), dest=outer_send_rank
                    )
                    outer_receive_request = comm1.Irecv(
                        buf=(buffer.get_receive_buffer(), MPI_DTYPE), source=outer_receive_rank
                    )     

                C_curr = get_subtile(C, px, 1, C_index, 0)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, px, 1, C_index, 0)     

                if j != inner_loop_iterations - 1:
                    MPI.Request.Waitall([inner_send_request, inner_receive_request])
                    buffer.swap()
                elif i != outer_loop_iterations - 1:
                    MPI.Request.Waitall([outer_send_request, outer_receive_request])
                    buffer.swap()

                # WHAT IS THIS IF STATEMENT
                if j != inner_loop_iterations - 1:
                    B_index = (B_index + 1) % py

            C_index = (C_index + 1) % px

        return C

    comm.Barrier()
    start_time = MPI.Wtime()
    C_local = algorithm(A_local, B_local, C_local, A_comm1, A_comm2, px, py)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    comm.Barrier()

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    output = {
        "elapsed_time": elapsed_time,
        "correct": correct,
        "matrices": {
            "A": A,
            "B": B,
            "C": C
        },
        "expected": expected,
        "actual": actual
    }

    return output


def AG_A_COL_AG_B_COL(m, k, n, px, py):
    # 2
    np.random.seed(43)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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
        outer_buffer = DoubleBuffer(A)
        inner_loop_iterations = px
        inner_buffer = np.empty(shape=get_subtile_shape(B, py, 1, B_index, 0))

        for i in range(outer_loop_iterations):

            if i != outer_loop_iterations - 1:
                outer_send_rank = (comm1_rank - 1) % comm1.Get_size()
                outer_receive_rank = (comm1_rank + 1) % comm1.Get_size()
                outer_send_request = comm1.Isend(
                    buf=(np.ascontiguousarray(outer_buffer.get_current_tile()), MPI_DTYPE), dest=outer_send_rank
                )
                outer_receive_request = comm1.Irecv(
                    buf=(outer_buffer.get_receive_buffer(), MPI_DTYPE), source=outer_receive_rank
                )

            for j in range(inner_loop_iterations):
                
                A_curr = outer_buffer.get_current_tile()
                B_curr = get_subtile(B, py, 1, B_index, 0)

                if j != inner_loop_iterations - 1:
                    inner_send_rank = (comm2_rank - 1) % comm2.Get_size()
                    inner_receive_rank = (comm2_rank + 1) % comm2.Get_size()
                    inner_send_request = comm2.Isend(
                        buf=(np.ascontiguousarray(B_curr), MPI_DTYPE), dest=inner_send_rank
                    )
                    inner_receive_request = comm2.Irecv(
                        buf=(inner_buffer, MPI_DTYPE), source=inner_receive_rank
                    )

                C_curr = get_subtile(C, 1, px, 0, C_index)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, 1, px, 0, C_index)

                if j != inner_loop_iterations - 1:
                    MPI.Request.Waitall([inner_send_request, inner_receive_request])
                    set_subtile(B, inner_buffer, py, 1, B_index, 0)

                C_index = (C_index + 1) % px

            if i != outer_loop_iterations - 1:
                MPI.Request.Waitall([outer_send_request, outer_receive_request])
                outer_buffer.swap()

            B_index = (B_index + 1) % py

        return C

    
    comm.Barrier()
    start_time = MPI.Wtime()
    C_local = algorithm(A_local, B_local, C_local, A_comm, B_comm, px, py)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    comm.Barrier()

    actual_tiles = comm.allgather((C_local, col_major_distribution_get_local_indices(px, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    output = {
        "elapsed_time": elapsed_time,
        "correct": correct,
        "matrices": {
            "A": A,
            "B": B,
            "C": C
        },
        "expected": expected,
        "actual": actual
    }

    return output

def AG_A_COL_AG_B_ROW(m, k, n, px, py):
    # 3
    np.random.seed(42)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert m % px == 0
    assert k % size == 0
    assert n % py == 0

    A, B, C = generate_matrices(m, k ,n)

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
        outer_buffer = np.empty(shape=get_subtile_shape(B, py, 1, B_index, 0))#DoubleBuffer(get_subtile(B, py, 1, outer_index, 0))
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(A, 1, px, 0, A_index))#DoubleBuffer(get_subtile(A, 1, px, 0, inner_index))


        for i in range(outer_loop_iterations):

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, 1, px, 0, A_index)

                if j != inner_loop_iterations - 1:
                    inner_send_rank = (comm2_rank - 1) % comm2.Get_size()
                    inner_receive_rank = (comm2_rank + 1) % comm2.Get_size()
                    inner_send_request = comm2.Isend(
                        buf=(np.ascontiguousarray(A_curr), MPI_DTYPE), dest=inner_send_rank
                    )
                    inner_receive_request = comm2.Irecv(
                        buf=(inner_buffer, MPI_DTYPE), source=inner_receive_rank
                    )

                B_curr = get_subtile(B, py, 1, B_index, 0)

                if i != outer_loop_iterations - 1:
                    outer_send_rank = (comm1_rank - 1) % comm1.Get_size()
                    outer_receive_rank = (comm1_rank + 1) % comm1.Get_size()
                    outer_send_request = comm1.Isend(
                        buf=(np.ascontiguousarray(B_curr), MPI_DTYPE), dest=outer_send_rank
                    )
                    outer_receive_request = comm1.Irecv(
                        buf=(outer_buffer, MPI_DTYPE), source=outer_receive_rank
                    )

                C = np.matmul(A_curr, B_curr) + C

                if j != inner_loop_iterations - 1:
                    MPI.Request.Waitall([inner_send_request, inner_receive_request])
                    set_subtile(A, inner_buffer, 1, px, 0, A_index)

                if i != outer_loop_iterations - 1:
                    MPI.Request.Waitall([outer_send_request, outer_receive_request])
                    set_subtile(B, outer_buffer, py, 1, B_index, 0)

                B_index = (B_index + 1) % py

            A_index = (A_index + 1) % px

        return C


    comm.Barrier()
    start_time = MPI.Wtime()
    C_local = algorithm(A_local, B_local, C_local, B_comm, A_comm, px, py)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    comm.Barrier()

    expected = np.matmul(A, B) + C    
    actual_tiles = comm.allgather((C_local, row_major_distribution_get_local_indices(py, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    output = {
        "elapsed_time": elapsed_time,
        "correct": correct,
        "matrices": {
            "A": A,
            "B": B,
            "C": C
        },
        "expected": expected,
        "actual": actual
    }

    return output

    

def AG_A_COL_RS_C_COL(m, k, n, px, py):
    # 4
    pass

def AG_A_COL_RS_C_ROW(m, k, n, px, py):
    # 5
    pass

def AG_A_ROW_AG_B_COL(m, k, n, px, py):
    # 6
    pass

def AG_A_ROW_AG_B_ROW(m, k, n, px, py):
    # 7
    np.random.seed(42)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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
        outer_buffer = DoubleBuffer(B)
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(A, 1, px, 0, A_index))

        for i in range(outer_loop_iterations):
            
            B_curr = outer_buffer.get_current_tile()

            if i != outer_loop_iterations - 1:
                outer_send_rank = (comm1_rank - 1) % comm1.Get_size()
                outer_receive_rank = (comm1_rank + 1) % comm1.Get_size()
                outer_send_request = comm1.Isend(
                    buf=(np.ascontiguousarray(outer_buffer.get_current_tile()), MPI_DTYPE), dest=outer_send_rank
                )
                outer_receive_request = comm1.Irecv(
                    buf=(outer_buffer.get_receive_buffer(), MPI_DTYPE), source=outer_receive_rank
                )  

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, 1, px, 0, A_index)

                if j != inner_loop_iterations - 1:
                    inner_send_rank = (comm2_rank - 1) % comm2.Get_size()
                    inner_receive_rank = (comm2_rank + 1) % comm2.Get_size()
                    inner_send_request = comm2.Isend(
                        buf=(np.ascontiguousarray(A_curr), MPI_DTYPE), dest=inner_send_rank
                    )
                    inner_receive_request = comm2.Irecv(
                        buf=(inner_buffer, MPI_DTYPE), source=inner_receive_rank
                    ) 

                C_curr = get_subtile(C, py, 1, C_index, 0)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, py, 1, C_index, 0)

                if j != inner_loop_iterations - 1:
                    MPI.Request.Waitall([inner_send_request, inner_receive_request])
                    set_subtile(A, inner_buffer, 1, px, 0, A_index)   

                C_index = (C_index + 1) % py       

            if i != outer_loop_iterations - 1:
                MPI.Request.Waitall([outer_send_request, outer_receive_request])
                outer_buffer.swap()       

            A_index = (A_index + 1) % px
        
        return C

    comm.Barrier()
    start_time = MPI.Wtime()
    C_local = algorithm(A_local, B_local, C_local, B_comm, A_comm, px, py)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    comm.Barrier()

    actual_tiles = comm.allgather((C_local, row_major_distribution_get_local_indices(py, rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    output = {
        "elapsed_time": elapsed_time,
        "correct": correct,
        "matrices": {
            "A": A,
            "B": B,
            "C": C
        },
        "expected": expected,
        "actual": actual
    }

    return output

def AG_A_ROW_RS_C_COL(m, k, n, px, py):
    # 8
    np.random.seed(42)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert m % px == 0
    assert k % py == 0
    assert n % size == 0

    A, B, C = generate_matrices(m, k ,n)
    expected = np.matmul(A, B) + C

    A_comm = remainder_communicator(comm, py, rank)
    C_comm = nearby_rank_communicator(comm, py, rank)

    A_local = row_major_distribution(A, px, py, rank)
    B_local = col_major_distribution(B, py, px, rank) # flippped!
    C_local = pure_column_distribution(C, size, rank) 

    def algorithm(A, B, C, comm1, comm2, px, py):
        comm1_rank = comm1.Get_rank()
        comm2_rank = comm2.Get_rank()
        B_index = (comm2_rank + 1) % py
        C_index = comm1_rank
        outer_loop_iterations = px
        outer_buffer = DoubleBuffer(A)
        inner_loop_iterations = py
        inner_buffer = np.empty(shape=get_subtile_shape(C, px, 1, C_index, 0))

        for i in range(outer_loop_iterations):

            A_curr = outer_buffer.get_current_tile()

            if i != outer_loop_iterations - 1:
                outer_send_rank = (comm1_rank - 1) % comm1.Get_size()
                outer_receive_rank = (comm1_rank + 1) % comm1.Get_size()
                outer_send_request = comm1.Isend(
                    buf=(np.ascontiguousarray(outer_buffer.get_current_tile()), MPI_DTYPE), dest=outer_send_rank
                )
                outer_receive_request = comm1.Irecv(
                    buf=(outer_buffer.get_receive_buffer(), MPI_DTYPE), source=outer_receive_rank
                ) 

            for j in range(inner_loop_iterations):

                B_curr = get_subtile(B, 1, py, 0, B_index)

                if j == 0:
                    C_curr = np.zeros(shape=get_subtile_shape(C, px, 1, C_index, 0))            
                else:
                    MPI.Request.Waitall([inner_receive_request, inner_send_request])
                    C_curr = inner_buffer

                C_curr = np.matmul(A_curr, B_curr) + C_curr

                if j == inner_loop_iterations - 1:
                    C_curr = C_curr + get_subtile(C, px, 1, C_index, 0)   
                    set_subtile(C, C_curr, px, 1, C_index, 0)
                else:
                    inner_send_rank = (comm2_rank - 1) % comm2.Get_size()
                    inner_receive_rank = (comm2_rank + 1) % comm2.Get_size()
                    inner_send_request = comm2.Isend(
                        buf=(np.ascontiguousarray(C_curr), MPI_DTYPE), dest=inner_send_rank
                    )
                    inner_receive_request = comm2.Irecv(
                        buf=(inner_buffer, MPI_DTYPE), source=inner_receive_rank
                    )

                B_index = (B_index + 1) % py 

            if i != outer_loop_iterations - 1:
                MPI.Request.Waitall([outer_send_request, outer_receive_request])
                outer_buffer.swap() 

            C_index = (C_index + 1) % px
        
        return C


    comm.Barrier()
    start_time = MPI.Wtime()
    C_local = algorithm(A_local, B_local, C_local, A_comm, C_comm, px, py)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    comm.Barrier()

    actual_tiles = comm.allgather((C_local, pure_column_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    output = {
        "elapsed_time": elapsed_time,
        "correct": correct,
        "matrices": {
            "A": A,
            "B": B,
            "C": C
        },
        "expected": expected,
        "actual": actual
    }

    return output       

def AG_A_ROW_RS_C_ROW(m, k, n, px, py):
    # 9
    pass

def AG_B_COL_AG_B_ROW(m, k, n, px, py):
    # 10
    np.random.seed(42)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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
        buffer = DoubleBuffer(B)
        inner_loop_iterations = px

        for i in range(outer_loop_iterations):

            for j in range(inner_loop_iterations):

                A_curr = get_subtile(A, 1, px, 0, A_index)
                B_curr = buffer.get_current_tile()

                if j != inner_loop_iterations - 1:
                    inner_send_rank = (comm2_rank - 1) % comm2.Get_size()
                    inner_receive_rank = (comm2_rank + 1) % comm2.Get_size()
                    inner_send_request = comm2.Isend(
                        buf=(np.ascontiguousarray(buffer.get_current_tile()), MPI_DTYPE), dest=inner_send_rank
                    )
                    inner_receive_request = comm2.Irecv(
                        buf=(buffer.get_receive_buffer(), MPI_DTYPE), source=inner_receive_rank
                    )
                elif i != outer_loop_iterations - 1:
                    outer_send_rank = (comm1_rank - 1) % comm1.Get_size()
                    outer_receive_rank = (comm1_rank + 1) % comm1.Get_size()
                    outer_send_request = comm1.Isend(
                        buf=(np.ascontiguousarray(buffer.get_current_tile()), MPI_DTYPE), dest=outer_send_rank
                    )
                    outer_receive_request = comm1.Irecv(
                        buf=(buffer.get_receive_buffer(), MPI_DTYPE), source=outer_receive_rank
                    )  

                C_curr = get_subtile(C, 1, py, 0, C_index)

                # print_local_matrices_on_debug_rank(A_curr, B_curr, C_curr)

                C_tmp = np.matmul(A_curr, B_curr) + C_curr
                set_subtile(C, C_tmp, 1, py, 0, C_index)

                if j != inner_loop_iterations - 1:
                    MPI.Request.Waitall([inner_send_request, inner_receive_request])
                    buffer.swap()
                elif i != outer_loop_iterations - 1:
                    MPI.Request.Waitall([outer_send_request, outer_receive_request])
                    buffer.swap()

                if j != inner_loop_iterations - 1:
                    A_index = (A_index + 1) % px

            C_index = (C_index + 1) % py
        
        return C

    # print_full_matrices(A, B, C)
    # print_local_matrices_on_debug_rank(A_local, B_local, C_local)

    comm.Barrier()
    start_time = MPI.Wtime()
    C_local = algorithm(A_local, B_local, C_local, B_comm1, B_comm2, px, py)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    comm.Barrier() 

    actual_tiles = comm.allgather((C_local, pure_row_distribution_get_local_indices(rank)))
    actual = assemble_matrix_from_tiles(actual_tiles)

    correct = matrices_equal(expected, actual)

    output = {
        "elapsed_time": elapsed_time,
        "correct": correct,
        "matrices": {
            "A": A,
            "B": B,
            "C": C
        },
        "expected": expected,
        "actual": actual
    }

    return output  

def AG_B_COL_RS_C_COL(m, k, n, px, py):
    # 11
    pass

def AG_B_COL_RS_C_ROW(m, k, n, px, py):
    # 12
    pass

def AG_B_ROW_RS_C_COL(m, k, n, px, py):
    # 13
    pass

def AG_B_ROW_RS_C_ROW(m, k, n, px, py):
    # 14
    pass

def RS_C_COL_RS_C_ROW(m, k, n, px, py):
    # 15
    pass
