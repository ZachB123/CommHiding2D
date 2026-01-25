def nearby_rank_communicator(comm, size_of_groups, rank):
    # GROUPS THE NEARBY RANK NUMBERS TOGETHER
    # use if you want like 0,1 2,3 4,5 to be groups etc
    # the communicator is a group that contains a chunk of a column
    # communicates across rows
    # used in 3
    return comm.Split(rank // size_of_groups, rank)


def remainder_communicator(comm, num_groups, rank):
    # same remainder so like if you have columns but order row major or rows but column major
    return comm.Split(rank % num_groups, rank)