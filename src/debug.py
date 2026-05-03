from mpi4py import MPI

from constants import DEBUG_RANK


def parallel_print(message, flush=True):
    """
    Print a message with the MPI rank and total size, color-coded by rank.

    Args:
        message (str): The message to print.
        flush (bool, optional): Whether to flush the output immediately (default is True).
    """
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    def get_color_code(rank, num_colors):
        return f"\033[38;5;{rank % num_colors}m"

    color_code = get_color_code(rank, size)

    print(f"{color_code}[{rank}/{size - 1}]\n{message}\033[0m", flush=flush)


def rank_print(message, print_rank=DEBUG_RANK, flush=True):
    """
    Print a message from a specific MPI rank.

    Args:
        message (str): The message to print.
        print_rank (int, optional): The rank that is allowed to print the message (default is 0).
        flush (bool, optional): Whether to flush the output immediately (default is True).
    """
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if print_rank >= size:
        if rank == 0:
            print(message, flush=flush)
    elif rank == print_rank:
        print(message, flush=flush)


def print_full_matrices(A, B, C):
    debug_string = f"A:\n{A}\nB:\n{B}\nC:\n{C}"
    rank_print(debug_string)


def print_ranks(comm1, comm2):
    debug_string = f"comm1: {comm1.Get_rank()}, comm2: {comm2.Get_rank()}"
    parallel_print(debug_string)


def print_local_matrices(A, B, C, start=None):
    debug_string = f"A:\n{A}\nB:\n{B}\nC:\n{C}\n"
    if start is not None:
        debug_string = f"{start}\n{debug_string}"
    parallel_print(debug_string)


def print_local_matrices_on_debug_rank(A, B, C, start=None, debug_rank=DEBUG_RANK):
    debug_string = f"A:\n{A}\nB:\n{B}\nC:\n{C}\n"
    if start is not None:
        debug_string = f"{start}\n{debug_string}"
    rank_print(debug_string, debug_rank)