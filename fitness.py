'''Fitness evaluation functions and related utilities.
'''
from config import FitnessConfig


# The fitness goals to run on all experiments, to get a rough estimate of their
# performance. We avoid running all goals on all configuratinos to reduce the
# total training time.
DEFAULT_GOALS = (FitnessConfig.RUN_LENGTH,)
# All fitness goals, to thoroughly evaluate the performance of a single
# configuration.
ALL_GOALS = (FitnessConfig.RUN_LENGTH,)


# All experiments should run for approximately this many steps.
TARGET_RUN_LENGTH = 100


def run_length(organism):
    '''Prefer simulations that run for the TARGET_RUN_LENGTH.
    '''
    # If the organism never reached a steady state, consider it invalid.
    if organism.last_step is None:
        return 0

    # Return a score from 0 to 100 indicating how close the organism was to
    # running the desired number of steps.
    return 100 - abs(TARGET_RUN_LENGTH - organism.last_step)


def evaluate_fitness(organism):
    '''Invoke the correct fitness function for the current experiment.
    '''
    fitness_config = organism.experiment_config.fitness_config
    return {
        FitnessConfig.RUN_LENGTH: run_length,
    }[fitness_config](organism)
