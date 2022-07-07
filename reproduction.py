'''Manage reproduction of organisms.
'''

from copy import deepcopy
from genome import Genotype


def randomize_crossover(reproduction_config):
    '''Decide whether or not to perform crossover in this breeding.
    '''
    # TODO: Decide randomly based on crossover rate.
    return False


def remix(parent_genes, mate_genes):
    '''Perform crossover by recombining two sets of genes.
    '''
    # TODO: Actually recombine genes from both parents.
    return deepcopy(parent_genes)


def mutate(genes):
    '''Add some random noise to the given genes.
    '''
    # TODO: Actually mutate the genotype.


def make_genotype(reproduction_config, parent, mate, logger):
    '''Construct a new genotype from parent.

    The parent produces one offspring, optionally doing crossover with mate.
    '''
    parent_fitness = parent.fitness
    grandparent_fitness = parent.genotype.parent_fitness
    fitness_vector = grandparent_fitness - parent_fitness
    do_crossover = (parent is not mate and
                    randomize_crossover(reproduction_config))
    if do_crossover:
        genes = remix(parent.genotype.genes, mate.genotype.genes)
    else:
        genes = deepcopy(parent.genotype.genes)
    mutate(genes)
    offspring = Genotype(parent_fitness, fitness_vector, genes)
    logger.log_breeding(offspring.identifier, parent.identifier,
                        mate.identifier, do_crossover)
    return offspring
