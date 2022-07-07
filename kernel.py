'''CUDA kernels for generating and running Game of Life simulations.

This is basically C code written with Python syntax, which is how Numba does
things. I considered just writing the kernels in C, but it seemed better to
avoid adding a second programming language to the project.
'''
from math import ceil


# Memory model for these CUDA kernels:
#
#     +-------+-------+
#   +-------+-------+ |
# +-------+-------+ |-+
# | 32x32 | 32x32 |-+ |
# +-------+-------+ |-+
# | 32x32 | 32x32 |-+
# +-------+-------+
#
# World size = 64x64
# Each world is a 2x2 array of blocks with 32 threads, one per cell
# Z dimension for blocks is parallel worlds (32 total)
# 32 * (2 * 2) == 128 blocks == 100% capacity

# The size of one side of the a square simulated world
WORLD_SIZE = 64

# The dimensions for a square world.
WORLD_SHAPE = (WORLD_SIZE, WORLD_SIZE)

# We'll allocate blocks and threads in square layouts.
# 32 x 32 == 1024 == max threads per block on this device.
THREAD_SIZE = 32

# Tile the given world size with blocks
BLOCK_SIZE = ceil(WORLD_SIZE / THREAD_SIZE)  # == 2

# The maximum number of blocks available on this device.
MAX_BLOCKS = 128

# The number of simulations to run in parallel.
NUM_SIMS = int(MAX_BLOCKS / (BLOCK_SIZE * BLOCK_SIZE))  # == 32

# States for cells in the world.
ALIVE = 0
DEAD = 255

# Gene keys for phenotype
RAW_SEED = 'raw_seed'


def step_simulations():
    '''Update all simulation worlds to the next time step.
    '''
    # TODO: Update all the simulations by one step in parallel.


def make_phenotypes(logger):
    '''Initialize simulation worlds by constructing phenotypes from genotypes.
    '''
    # TODO: Generate the first frame for each of the simulations in parallel by
    # transforming each organism's genotype to a phenotype. Use the logger to
    # record the process.
