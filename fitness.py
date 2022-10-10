"""Fitness evaluation functions and related utilities.

This module provides several predefined fitness goals for use with the
GameOfLifeSimulation class in the EXPERIMENT_GOALS dictionary. Each goal is a
function that takes a GameOfLifeSimulation and returns a fitness score. This
score is an integer with an arbitrary scale that is meant for comparing
relative performance on a single goal, but not comparing performance between
goals.
"""
import functools

import numpy as np

import kernel


# A dict containing all the fitness goals to use for this project's
# experiments. The keys to this dict are the names of each fitness goal and the
# values are corresponding functions that take a GameOfLifeSimulation video as
# their parameter and return an integer fitness score. This is populated when
# this module is loaded via the fitness_goal decorator.
EXPERIMENT_GOALS = {}


def get_frames_needed(fitness_function):
    """Look up the frames needed to evaluate a fitness goal specified by name.

    This is used to improve performance by avoiding unnecessary memory
    transfers from the GPU (see documentation in the gol_simulation module).

    Parameters
    ----------
    fitness_function : Callable
        A fitness function declared using the @fitness_goal decorator.

    Result
    ------
    list of int
        A list of frame indices analyzed by the fitness function. This is used
        to avoid capturing every frame of a GOL simulation, which would badly
        hurt performance. This return value is meant for use as a parameter to
        gol_simulation.simulate.
    """
    # This metadata is attached to the function object by the fitness_goal
    # decorator.
    return fitness_function.frames_needed


def evaluate_simulation(fitness_function, simulation):
    """Evaluate the fitness of a simulation given a specific fitness function.

    This function serves as an adapter from a GameOfLifeSimulation to just the
    simulation video frames needed by one of the below fitness functions. You
    probably have no need to call this function directly. The EXPERIMENT_GOALS
    dictionary is more convenient, and its values are just instances of this
    function with the first argument bound to the appropriate fitness function
    for each key / fitness goal name.

    Parameters
    ----------
    fitness_function : Callabale
        A fitness function declared using the @fitness_goal decorator.
    simulation : GameOfLifeSimulation
        The simulation to evaluate the fitness of. This simulation should
        already have been run with the relevant frames (as determined by
        get_frames_needed) recorded by gol_simulation.simulate.

    Returns
    -------
    int
        A fitness score on an arbitrary scale. Higher values are better.
    """
    frames_needed = get_frames_needed(fitness_function)
    frames = [simulation.frames[i] for i in frames_needed]
    return fitness_function(*frames)


def fitness_goal(frames_needed):
    """A decorator for convenient definition of fitness goals.

    The purpose of this decorator is to bind together a fitness function with a
    name and a set of frames needed to compute the fitness score without a
    bunch of awkward boilerplate or repetition. It populates the DEFAULT_GOALS
    dictionary which is the primary entrypoint for this module.

    Parameters
    ----------
    frames_needed : list of int
        A list of frame indices analyzed by the fitness function. This is used
        to avoid capturing every frame of a GOL simulation, which would badly
        hurt performance. The values in this list should be relative indices
        from the start or the end of the simulation video. These will be
        expanded as the named arguments of the function.
    """
    # We need to pass frames_needed as an argument to fitness_goal, but a
    # decorator must be a function that takes a function as a parameter. So,
    # calling fitness_goal evaluates to a decorator that takes the actual
    # goal-specific fitness function as its parameter.
    def decorator(fitness_function):
        # The name of the function being wrapped is used as its identifier when
        # analyzing or outputing experiment results. It's also the key for
        # looking up this fitness_function in EXPERIMENT_GOALS.
        goal_name = fitness_function.__name__
        # Wrap the fitness function using the evaluate_simulation function to
        # provide a convenient interface to callers.
        wrapped_function = functools.partial(
            evaluate_simulation, fitness_function)
        EXPERIMENT_GOALS[goal_name] = wrapped_function
        # Track the frames needed list by attaching it to the function. This
        # seemed simpler than creating a parallel global dict for metadata.
        # Attach it both to the wrapped and unwrapped versions, so that
        # get_frames_needed doesn't have to care which its called with.
        fitness_function.frames_needed = frames_needed
        wrapped_function.frames_needed = frames_needed
        # This decorator just tracks metadata for the given fitness_function,
        # but does not modify its behavior. So, calling the decorator evaluates
        # to the decorated function.
        return fitness_function
    return decorator


@fitness_goal([-2, -1])
def active(prev_frame, last_frame):
    """Number of cells changed between the last two frames."""
    return np.count_nonzero(last_frame != prev_frame)


@fitness_goal([-2, -1])
def still_life(prev_frame, last_frame):
    """Number of live cells that are the same between the last two frames."""
    return np.count_nonzero(np.logical_and(
        last_frame == prev_frame, last_frame == kernel.ALIVE))


@fitness_goal([0, -1])
def explode(first_frame, last_frame):
    """Relative increase in living cells between first and last frames."""
    alive_on_first = np.count_nonzero(first_frame == kernel.ALIVE)
    alive_on_last = np.count_nonzero(last_frame == kernel.ALIVE)
    return int(100 * alive_on_last / (1 + alive_on_first))


@fitness_goal([-1])
def full(last_frame):
    """Number of cells alive in the last frame."""
    return np.count_nonzero(last_frame == kernel.ALIVE)


@fitness_goal([-4, -3, -2, -1])
def two_cycle(prev_even, prev_odd, last_even, last_odd):
    """Number of cells oscillating between two values at the end."""
    # Find cells which had the same values in the odd / even numbered frames.
    odd_same = last_odd == prev_odd
    even_same = last_even == prev_even
    # Find cells which stayed the same across odd and even numbered frames.
    static = last_odd == last_even
    # Count the number of cells that were the same two frames apart but were
    # not the same from frame to frame.
    return np.count_nonzero(
        np.logical_and(odd_same == even_same, np.logical_not(static)))


@fitness_goal([-6, -5, -4, -3, -2, -1])
def three_cycle(prev_a, prev_b, prev_c, last_a, last_b, last_c):
    """Number of cells oscillating between three values at the end."""
    # Find cells that stayed the same on frames three steps apart.
    a_same = last_a == prev_a
    b_same = last_b == prev_b
    c_same = last_c == prev_c
    # Find cells that stayed the same across all six frames.
    static = np.logical_and(last_a == last_b, last_b == last_c)
    # Find cells that had consistent values in frames three steps apart but
    # where not the same from frame to frame.
    cycling = np.logical_and(
        a_same, np.logical_and(
            b_same, np.logical_and(
                c_same, np.logical_not(static))))
    # Count up how many cells fit that criteria.
    return np.count_nonzero(cycling)


@fitness_goal([0, -1])
def left_to_right(first_frame, last_frame):
    """Concentration of living cells shifts from left to right.

    This fitness goal finds the ratio of living cells from left to right on
    both the first frame and the last frame of the simulation. The fitness
    score is the product of these two values, so that it isn't possible for a
    really good first frame to overwhelm the contribution from a really bad
    last frame, or vice versa.
    """
    # Split the first and last frames into left and right halves.
    first_left, first_right = np.split(first_frame, 2, axis=1)
    last_left, last_right = np.split(last_frame, 2, axis=1)
    # Count the ratio of live cells on the left / right sides in the first
    # frame and then again for the second frame.
    first_ratio = (np.count_nonzero(first_left == kernel.ALIVE) /
                   (np.count_nonzero(first_right == kernel.ALIVE) + 1))
    last_ratio = (np.count_nonzero(last_right == kernel.ALIVE) /
                  (np.count_nonzero(last_left == kernel.ALIVE) + 1))
    # Combine the two ratios into one fitness score.
    return first_ratio * last_ratio


@fitness_goal([-1])
def symmetry(last_frame):
    """Most cells with vertical or horizontal symmetry in the last frame."""
    # Split the last frame into halves and flip the bottom to see if it's a
    # mirror of the top.
    top_half, bottom_half = np.split(last_frame, 2, axis=0)
    bottom_half = np.flip(bottom_half, axis=0)
    # Do the same for the left and right halves.
    left_half, right_half = np.split(last_frame, 2, axis=1)
    right_half = np.flip(right_half, axis=1)
    # Count up how many live pixels have the same value when mirror
    # horizontally and vertically. Note we only count live cells, so degenerate
    # cases like an empty game board don't get a high score.
    vsimilarity = np.count_nonzero(np.logical_and(
        top_half == kernel.ALIVE, bottom_half == kernel.ALIVE))
    hsimilarity = np.count_nonzero(np.logical_and(
        left_half == kernel.ALIVE, right_half == kernel.ALIVE))
    # Combine the two ratios into one fitness score.
    return vsimilarity + hsimilarity


# TODO: It would be really nice to have fitness goals looking for specific
# still lifes, oscillators, and spaceships, or attempting to find as many novel
# patterns as possible. Those would be more difficult to implement than the
# goals here, so are left as future work.
