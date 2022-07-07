'''Simulate and evolve populations of organisms in parallel simulations.
'''

import random
import time

from gol_simulation import GameOfLifeSimulation
import fitness
import kernel

MIN_RUN_LENGTH = fitness.TARGET_RUN_LENGTH
MAX_RUN_LENGTH = int(1.5 * MIN_RUN_LENGTH)
NUM_GENERATIONS = 100


def simulate(organisms, logger):
    '''Simulate the lives of the given organisms.
    '''
    # TODO: Support batching.
    assert len(organisms) <= kernel.NUM_SIMS
    simulator = MultiverseSimulator()
    for organism in organisms:
        simulator.allocate(organism)
    kernel.make_phenotypes(logger)
    simulator.run_simulations(logger)


class MultiverseSimulator:
    '''A class for running parallel organism simulations.
    '''
    def __init__(self):
        self.organisms = []
        self.best_organism = None

    def allocate(self, organism):
        '''Assign this organism to one slot in the multiverse simulator.
        '''
        organism.sim_index = len(self.organisms)
        self.organisms.append(organism)

    def populate(self, genotypes, experiment_config, logger):
        '''Constructs a new generation of organisms from genotypes.
        '''
        self.organisms.clear()
        for genotype in genotypes:
            organism = GameOfLifeSimulation(genotype, experiment_config)
            self.allocate(organism)
        kernel.make_phenotypes(logger)

    def run_simulations(self, logger):
        '''Run the GOL simulation for all organisms.

        This method will run all simulations for at least MIN_RUN_LENGTH steps,
        but will terminate early if all simulations have settled into a steady
        state.
        '''
        for organism in self.organisms:
            organism.before_run()
        for step in range(MAX_RUN_LENGTH):
            for organism in self.organisms:
                organism.before_step(logger)
            kernel.step_simulations()
            all_stopped = True
            for organism in self.organisms:
                organism.after_step(step)
                if organism.last_step is None:
                    all_stopped = False
            if step >= MIN_RUN_LENGTH and all_stopped:
                break
        for organism in self.organisms:
            organism.after_run()

    def get_best_from_population(self):
        '''Finds the most fit organism in the current population.

        This method also has the side effect of keeping track of the best
        organism across all calls in self.best_organism.
        '''
        best_of_this_population = None
        for organism in self.organisms:
            if organism > best_of_this_population:
                best_of_this_population = organism
        if best_of_this_population > self.best_organism:
            self.best_organism = best_of_this_population
        return best_of_this_population

    def select(self, count):
        '''Choose count organisms from the population in proportion to fitness.

        This method uses Stochastic Universal Sampling. This randomly selects
        organisms proportionate to fitness, but avoids statistically unlikely
        scenarios like choosing the same organism count times by basing all
        selections off of a single random value.
        '''
        # Imagine a roulette wheel, where each organism is assigned a wedge
        # that is as large as its fitness. The total circumference of that
        # wheel is the total fitness for the population.
        total_fitness = sum(organism.fitness for organism in self.organisms)
        # Pick count equidistant sampling points around the edge of that
        # roulette wheel, starting at a random location.
        sample_distance = total_fitness / count
        start_position = random.random() * sample_distance
        samples = [start_position + i * sample_distance for i in range(count)]

        # Visit each sample point and figure out which wedge of the roulette
        # wheel contains it. The organism corresponding to that wedge will be
        # selected.
        result = []
        index = 0
        fitness_so_far = 0
        for sample in samples:
            while sample > fitness_so_far:
                fitness_so_far += self.organisms[index].fitness
                index += 1
            result.append(self.organisms[index])
        return result

    def propagate(self, logger):
        '''Produce the next generation of genotypes from the previous one.
        '''
        population_size = len(self.organisms)
        parents = self.select(population_size)
        mates = self.select(population_size)
        random.shuffle(mates)
        genotypes = [parent.make_genotype(mate, logger)
                     for parent, mate in zip(parents, mates)]
        return genotypes

    def evolve(self, genotypes, experiment_config, logger):
        '''Evolve the population for NUM_GENERATIONS or until fitness levels off.
        '''
        # TODO: Support batching.
        assert len(genotypes) <= kernel.NUM_SIMS
        for generation in range(NUM_GENERATIONS):
            start_time = time.time()
            self.populate(genotypes, experiment_config, logger)
            self.run_simulations(logger)
            best_of_this_generation = self.get_best_from_population()
            # Don't bother propagating the next generation if this is the last
            # generation to run.
            if generation + 1 < NUM_GENERATIONS:
                genotypes = self.propagate(logger)
            elapsed_time = time.time() - start_time
            logger.log_generation(best_of_this_generation, elapsed_time)
