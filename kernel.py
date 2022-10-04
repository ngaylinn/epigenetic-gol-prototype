"""CUDA kernels for generating and running Game of Life simulations.

This code represents the inner loop of this project's evolutionary algorithm
experiments. It's transpiled into C on demand using Numba, then executes many
times in parallel on an NVidia GPU. This provides the necessary speed to make
these experiments practical. That means everything below is basically C code
written in a subset of Python syntax, which is gross. If this code gets much
longer, more complicated, or would benefit from access to more C features or
libraries, it might be better to replace this with a native C module.
"""

import math

from numba import cuda
import numpy as np


# Memory model for these CUDA kernels:
#
#     +-------+-------+
#   +-------+-------+ |
# +-------+-------+ |-+
# | 32x32 | 32x32 |-+ |
# +-------+-------+ |-+
# | 32x32 | 32x32 |-+
# +-------+-------+
# (stack of depth 32)
#
# Each Game of Life simulation is a grid of 64x64 cells. Each simulation is
# represented by a 2x2 array of blocks with 32x32 threads each, one per cell on
# the game board. The Z dimension for blocks represents parallel boards (32
# total) each running their own simulation.
#
# This code is optimized for an NVidia GPU with Compute Capability 8.x
# (See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
# So, 32 * (2 * 2) == 128 blocks, which is 100% utilization.

# The size of one side of the a square simulated world
WORLD_SIZE = 64

# The dimensions for a square world.
WORLD_SHAPE = (WORLD_SIZE, WORLD_SIZE)

# We'll allocate blocks and threads in square layouts.
# 32 x 32 == 1024 == max threads per block on this device.
THREAD_SIZE = 32

# Tile the given world size with blocks
BLOCK_SIZE = math.ceil(WORLD_SIZE / THREAD_SIZE)  # == 2

# The maximum number of blocks available on this device.
MAX_BLOCKS = 128

# The number of simulations to run in parallel.
NUM_SIMS = math.floor(MAX_BLOCKS / (BLOCK_SIZE * BLOCK_SIZE))  # == 32

# State values for cells in the world.
ALIVE = 0
DEAD = 255

# The size of a "stamp" for make_phenotype (see comments for genome.GENOME)
STAMP_SIZE = 8

# The dimensions of a stamp (referenced in the genome module)
STAMP_SHAPE = (STAMP_SIZE, STAMP_SIZE)

# Constants representing different stamp repeat modes (see comments for
# genome.GENOME)
REPEAT_NONE = 0
REPEAT_ONCE = 1
REPEAT_1D = 2
REPEAT_2D = 3

# Device-side arrays for running the simulation. The _device_frames array holds
# the last computed frame for each simulation, while _device_buffers is scratch
# space used to draw the next frame. After each kernel execution, these get
# swapped so the previous frame becomes the buffer for computing the next one.
_device_frames = cuda.device_array((NUM_SIMS,) + WORLD_SHAPE, np.uint8)
_device_buffers = cuda.device_array((NUM_SIMS,) + WORLD_SHAPE, np.uint8)

# A host-side copy of _device_frames, for transferring data from the GPU back
# to Python code running on the CPU. This variable is normally None and only
# gets updated on demand by calling get_frame to minimize memory transfers.
_host_frames = None
# A host-side array representing the genotypes for every simulation, for
# gathering data from the Python to be transferred to the GPU. This gets
# allocated by set_genotype and pushed to the GPU by make_phenotypes.
_host_genotypes = None


@cuda.jit
def in_bounds(row, col):
    """Check if a position is within the spacial bounds of the simulation."""
    return 0 <= row < WORLD_SIZE and 0 <= col < WORLD_SIZE


@cuda.jit
def _step_kernel(frames, buffers):
    # Each invocation of this function operates on a single cell (at position
    # row, col) within one of NUM_SIMs parallel simulations (indicated by
    # sim_index).
    row, col, sim_index = cuda.grid(3)
    if not in_bounds(row, col):
        return

    # Our input, the previous frame from the GOL simulation this thread is
    # working on.
    frame = frames[sim_index]
    # Our output, a space to put the world state that follows frame.
    buffer = buffers[sim_index]

    # Count how many of the neighboring cells are ALIVE
    neighbors = 0
    for row_off in (-1, 0, 1):
        for col_off in (-1, 0, 1):
            if (row_off, col_off) == (0, 0):
                continue
            if not in_bounds(row + row_off, col + col_off):
                continue
            if frame[row + row_off][col + col_off] == ALIVE:
                neighbors += 1

    # Compute the next state for this cell based on its previous state and its
    # number of living neighbors.
    last_state = frame[row][col]
    next_state = DEAD
    if last_state == ALIVE and (neighbors == 2 or neighbors == 3):
        next_state = ALIVE
    if last_state == DEAD and neighbors == 3:
        next_state = ALIVE
    buffer[row][col] = next_state


def step_simulations():
    """Computes the next frame for NUM_SIMS parallel GOL simulations.

    This function initializes the memory objects used by the GPU and then
    invokes _step_kernel to compute the next state for all the cells of all the
    simulations in paralel.

    This function assumes that the first frames for the simulations has already
    been initialized using make_phenotypes below.
    """
    global _device_frames, _device_buffers, _host_frames
    # Initialize the kernel dispatcher and launch the kernel. The kernel will
    # read the last frames from _device_frames and write the next computed
    # frames to _device_buffers.
    _step_kernel[
        # Layout blocks into grids
        (BLOCK_SIZE, BLOCK_SIZE, NUM_SIMS),
        # Layout threads into blocks
        (THREAD_SIZE, THREAD_SIZE)
    ](_device_frames, _device_buffers)
    # Clear the host-side frame cache since it no longer accurately reflects
    # the device-side copy. It can be updated on demand by calling get_frame.
    _host_frames = None
    # The data in buffer is now the last computed frame. The previous computed
    # frame is free to use as the buffer in the next iteration.
    _device_frames, _device_buffers = _device_buffers, _device_frames


@cuda.jit
def phenotype_kernel(genotypes, phenotypes):
    # Each invocation of this function operates on a single cell (at position
    # row, col) within one of NUM_SIMs parallel simulations (indicated by
    # sim_index).
    row, col, sim_index = cuda.grid(3)
    if not in_bounds(row, col):
        return

    # Unfortunately, Numba doesn't support structured arrays very well, so we
    # have to unpack the genotype fields manually, in the same order they are
    # defined in genome.GENOME.
    genotype = genotypes[sim_index]
    seed = genotype[0]
    stamp = genotype[1]
    stamp_offset_row = genotype[2][0]
    stamp_offset_col = genotype[2][1]
    repeat_mode = genotype[3]
    repeat_offset_row = genotype[4][0]
    repeat_offset_col = genotype[4][1]
    mirror = genotype[5]
    phenotype = phenotypes[sim_index]

    # If we're not using the stamp operation, then the phenotype is drawn
    # directly from the seed data.
    if not stamp:
        phenotype[row][col] = seed[row][col]
        return

    # At this point, we know we're drawing stamps. By default, assume the stamp
    # doesn't target this position and draw nothing.
    phenotype[row][col] = DEAD

    # A repeat_offset of (0, 0) is non-sensical and could cause division by
    # zero, so ignore it and don't attempt to repeat the stamp at all.
    if repeat_offset_row == 0 and repeat_offset_col == 0:
        repeat_mode = REPEAT_NONE
        # Compute the stamp-relative coordinates of row, col. If they are out
        # of bounds for the stamp, then draw nothing here.
        stamp_row = row - stamp_offset_row
        if stamp_row < 0 or stamp_row > STAMP_SIZE:
            return
        stamp_col = col - stamp_offset_col
        if stamp_col < 0 or stamp_col > STAMP_SIZE:
            return
    else:
        # Compute where (row, col) lies relative to the stamp, taking
        # repetitions into account. The variables stamp_row and stamp_col
        # represent stamp relative position. If those coordinates are out of
        # bounds of the stamp, draw nothing here. The variables repeat_row and
        # repeat_col indicate the number of horizontal or vertical repetitions
        # needed to draw a stamp in this position.
        repeat_row, stamp_row = divmod(
            row - stamp_offset_row, repeat_offset_row)
        if stamp_row < 0 or stamp_row > STAMP_SIZE:
            return
        repeat_col, stamp_col = divmod(
            col - stamp_offset_col, repeat_offset_col)
        if stamp_col < 0 or stamp_col > STAMP_SIZE:
            return

    # If repeat_mode is REPEAT_NONE, only draw the stamp once, not at any of
    # the possible repeat positions.
    if repeat_mode == REPEAT_NONE:
        if repeat_row != 0 or repeat_col != 0:
            return
    # If repeat_mode is REPEAT_ONCE, then draw two copies of the stamp, one at
    # stamp_offset + (0, 0) and another at stamp_offset + repeat_offset.
    elif repeat_mode == REPEAT_ONCE:
        if repeat_row > 1 or repeat_row < 0:
            return
        if repeat_col > 1 or repeat_col < 0:
            return
        if repeat_row != repeat_col:
            return
    # If repeat_mode is REPEAT_1D, then draw many copies of the stamp, in every
    # position that is stamp_offset plus some multiple of repeat_offset.
    elif repeat_mode == REPEAT_1D:
        if repeat_row != repeat_col:
            return
    # If repeat_mode is REPEAT_2D, then draw many copies of the stamp in a grid
    # layout with a distance of repeat_offset between stamps. In other words,
    # any value of repeat_row and repeat_col are acceptable.

    # If mirroring is enabled, then invert our stamp-relative coordinates for
    # every other copy of the stamp.
    if mirror and repeat_row % 2:
        stamp_row = STAMP_SIZE - stamp_row - 1
    if mirror and repeat_col % 2:
        stamp_col = STAMP_SIZE - stamp_col - 1

    # At this point, we've determined that the current row and column are
    # within the bounds of a copy of the stamp that we should draw, so we must
    # draw data from the stamp to output here. The source data for the stamp is
    # drawn from the middle of the seed (so it will get properly shuffled
    # during crossover), so we adjust the indices accordingly before reading
    # from the seed data.
    stamp_row = int(stamp_row + (WORLD_SIZE - 8) / 2)
    stamp_col = int(stamp_col + (WORLD_SIZE - 8) / 2)
    phenotype[row][col] = seed[stamp_row][stamp_col]


def make_phenotypes():
    """Initialize simulation worlds by constructing phenotypes from genotypes.

    This function initializes the memory objects used by the GPU and then
    invokes phenotype_kernel to compute the first frame for all the cells of
    all the simulations in paralel.

    Once this function has been called once, call step_simulations repeatedly
    to execut a GOL simulation.
    """
    global _device_frames, _device_buffers, _host_frames
    # Initialize the kernel dispatcher and launch the kernel. The kernel will
    # read genotype data from a device-side copy of _host_genotypes and use it
    # to draw the first frame of the simulation into _device_buffers.
    phenotype_kernel[
        # Layout blocks into grids
        (BLOCK_SIZE, BLOCK_SIZE, NUM_SIMS),
        # Layout threads into blocks
        (THREAD_SIZE, THREAD_SIZE)
    ](cuda.to_device(_host_genotypes), _device_buffers)
    # Clear the host-side frame cache since it no longer accurately reflects
    # the device-side copy. It can be updated on demand by calling get_frame.
    _host_frames = None
    # The data in buffer is now the first frame in the simulation, but
    # step_simulation expects it to be in _device_frames, so we swap the
    # pointers. We could just draw the phenotype directly to _device_frmaes,
    # but it seemed better to have a consistent convention of writing to
    # buffers and reading from frames.
    _device_frames, _device_buffers = _device_buffers, _device_frames


def get_frame(sim_index):
    """Grab a single frame from the simulation kernel.

    Normally the frames for a simulation live on the GPU, so we can compute
    frame after frame in quick succession without memory transfers. When we
    need to capture a frame to record a video or compute fitness, we copy the
    data from the GPU device. Although this function is called once per
    simulation, the memory transfer is a batch operation over all simulations.

    Parameters
    ----------
    sim_index : int
        Indicates which of the simulations to get the frame from.

    Returns
    -------
    np.ndarray of np.uint8
        The last computed frame of the indicated simulation. This is an array
        of bytes in the range 0 to 255 representing a grayscale image.
    """
    global _host_frames
    # _host_frames is set to None after each frame. When any one simulation
    # wants to capture a frame, we copy all the frames from the device in one
    # go and cache it until the next step is run. Usually this method is called
    # once for every sim_index in quick succession after a simulation step.
    if _host_frames is None:
        _host_frames = _device_frames.copy_to_host()
    return _host_frames[sim_index]


def set_genotype(sim_index, genotype_data):
    """Set the genotype for a simulation before computing its phenotype.

    All simulations must call this function before a call to make_phenotypes.

    Parameters
    ----------
    sim_index : int
        Indicates which simulation should use the given genotype.
    genotype_data : np.ndaryy
        The genotype data to use to make the phenotype for this simulation.
        This should normally be created by the Genotype class.
    """
    global _host_genotypes
    # This program assumes all individuals being simulated share the same
    # genome. On the first call to this function, allocate genotype space for
    # all the simulations to update in place for the rest of the program.
    if _host_genotypes is None:
        _host_genotypes = np.empty(NUM_SIMS, genotype_data.dtype)
    _host_genotypes[sim_index] = genotype_data
