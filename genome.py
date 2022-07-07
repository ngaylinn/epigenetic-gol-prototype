'''Model the genes for the GameOfLifeSimulation organism.
'''
import random
import numpy as np
import kernel


def _get_id_generator():
    next_id = 0
    while True:
        yield next_id
        next_id += 1


class Genotype:
    '''Collection of genes for a single organism.
    '''
    _id_generator = _get_id_generator()

    def __init__(self, parent_fitness, fitness_vector, genes):
        self.parent_fitness = parent_fitness
        self.fitness_vector = fitness_vector
        self.genes = genes
        self.identifier = next(self._id_generator)


# TODO: Actually explore all the genes.
def get_diverse_genotypes(reproduction_config, phenotype_config,
                          population_size):
    '''Generate a randomized and diverse collection of genotypes.
    '''
    result = []
    for index in range(population_size):
        density = (1 + index) / (1 + population_size)
        raw_seed = np.full(kernel.WORLD_SHAPE, kernel.DEAD, dtype=np.uint8)
        for row in range(kernel.WORLD_SIZE):
            for col in range(kernel.WORLD_SIZE):
                if random.random() < density:
                    raw_seed[row][col] = kernel.ALIVE
        genes = {
            kernel.RAW_SEED: raw_seed
        }
        result.append(Genotype(0, 0, genes))
    return result
