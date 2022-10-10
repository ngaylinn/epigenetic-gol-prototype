"""Infrastructure to run and collect data from a genetic algorithm experiment.

The purpose of this module is to separate the basic setup and state tracking
needed to run any genetic algorithm from the behavior of some specific genetic
algorithm. This infrastructre makes a few assumptions:
- The basic execution model is to start with an initial population,
  simulate and evaluate the fitness of each individual in that population,
  select a new population from the best of the previous, and repeat.
- The purpose is to find the most fit individuals and to track the best
  fitness score across generations.
"""

from functools import total_ordering
import random


def select(population, count):
    """Choose count individuals from population in proportion to fitness.

    This method uses Stochastic Universal Sampling. This randomly selects
    individuals proportionate to fitness, but avoids statistically unlikely
    scenarios (like choosing the same individual count times) by basing all
    selections off of a single random value.

    Parameters
    ----------
    population: list of Evolvable
        The pool of individuals to select from.
    count: int
        The number of individuals to select.
    """
    # Imagine a roulette wheel, where each individual is assigned a wedge
    # that is proportional its fitness. The full circumference of that
    # wheel is the total fitness for the population.
    total_fitness = sum(individual.fitness for individual in population)
    # Handle the degenerate case where everyone has fitness 0 by choosing
    # individuals at random.
    if total_fitness == 0:
        return random.sample(population, count)
    # Pick count equidistant sampling points around the edge of that
    # roulette wheel, starting at a random location.
    sample_period = total_fitness / count
    sample_offset = random.random() * sample_period
    samples = [sample_offset + i * sample_period for i in range(count)]

    # Walk around the edge of the roulette wheel to figure out which wedge
    # contains each sample point. The individual corresponding to that wedge /
    # sample point will be selected. More fit individuals have bigger wedges
    # which may contain multiple sample points, which means that individual
    # gets selected more than once and can have multiple offspring. Individuals
    # with a fitness score smaller than the sample_period may fall between
    # sample points and fail to get selected and pass on their genes.
    result = []
    # An index into population / the wedges in the roulette wheel. Starting at
    # -1 indicates we have not yet reached a sample point inside the 0th wedge,
    # but will do so in the first iteration of the loop below.
    index = -1
    fitness_so_far = 0
    for sample in samples:
        # Step through the wedges one at a time to find the one that overlaps
        # with this sample point. This could happen 0 times if the last wedge
        # is so big it contains this next sample point, too, or it could happen
        # many times, if there are many thin wedges to pass over.
        while sample > fitness_so_far:
            index += 1
            fitness_so_far += population[index].fitness
        # Actually select the individual corresponding to this sample point.
        result.append(population[index])
    return result


# Each Evolvable gets a tracking id, and this is a helper for generating
# sequential numbers for those ids.
def _get_id_generator():
    next_id = 0
    while True:
        yield next_id
        next_id += 1


_id_generators = {}


@total_ordering  # Sortable by fitness.
class Evolvable:
    """An individual evolvable by the Lineage class.

    This is an abstract base class meant to handle basic book keeping for each
    individual in a genetic algorithm, and provide an interface to Lineage.

    To subclass Evolvable, you must:
    - Call super().__init__ with an id prefix representing your subclass.
      Each individual will be assigned a unique id with that prefix.
    - Provide implementations of should_crossover and make_offspring.
    - Make sure to set fitness appropriately when implementing
      Lineage.evaluate_population.
    """
    def __init__(self, id_prefix):
        id_number = next(_id_generators.setdefault(
            id_prefix, _get_id_generator()))
        self.identifier = f'{id_prefix}{id_number:08d}'
        # To be set by subclasses
        self.fitness = None
        # Used by the debugger to visualize reproduction events.
        self.num_parents = 0
        self.num_children = 0

    def should_crossover(self):
        """Return true iff the make_offspring should use crossover.

        When true, this individual should reproduce sexually, combining its
        genes with those of a mate. Otherwise, it should reproduce asexually,
        producing offspring based on only its own genes.

        Returns
        -------
        bool
            True iff this breeding should use crossover.
        """
        raise NotImplementedError

    def make_offspring(self, mate=None):
        """Produce a new Evolvable using genes from self and possibly mate.

        The implementation for this method should apply both mutation and
        crossover. Traditionally, crossover is an operation that takes two
        genotypes and returns two genotypes, guaranteeing that all genes from
        both parents are present in the next generation. This project produces
        each offspring independently, which is more biologically realistic but
        also means valuable genes from the parent may not make it into the next
        generation.

        Parameters
        ----------
        mate : optional Evolvable
            Another individual from the same population that has been selected
            as a mate for this breeding. Note that if mate is provided, that
            means should_crossover returned True and crossover should be
            performed if applicable.

        Returns
        -------
        Evolvable
            The offspring of self, to include in the next population.
        """
        raise NotImplementedError

    def __eq__(self, other):
        return other is not None and self.fitness == other.fitness

    def __lt__(self, other):
        return other is not None and self.fitness < other.fitness


class Lineage:
    """Evolve an experimental population and capture statistics.

    This class handles the basic book keeping and statistics tracking of a
    genetic algorithm. To use it, make a subclass and implement
    make_initial_population and evaluate_population for your needs.

    Attributes
    ----------
    fitness_by_generation : list of int
        A list with the maximum fitness score from each generation.
    best_individual : Evolvable
        The single most fit individual found across all generations.
    best_individual_by_generation : list of Evolvable
        A list of the best individual from each generation.
    """
    def __init__(self):
        self.fitness_by_generation = []
        self.best_individual = None
        self.best_individual_by_generation = []
        # Hooks for the debugger to override. By default, they do nothing.
        self.per_generation_callback = lambda generation, population: None
        self.per_breeding_callback = lambda gen, parent, mate, child: None

    def make_initial_population(self):
        """Return a list of Evolvables to serve as the seed population.

        Returns
        -------
        list of Evolvable
            All the individuals for the first generation.
        """
        raise NotImplementedError

    def evaluate_population(self, population):
        """Evaluate all individuals from population and their fitness.

        When implementing this function, run the full life cycle of each
        individual in population and set individual.fitness to reflect their
        performance.

        Parameters
        ----------
        population : list of Evolvable
            All the individuals to evaluate.
        """
        raise NotImplementedError

    def run_generation(self, population):
        """Run a single generation and track fitness data.

        Parameters
        ----------
        population : list of Evolvable
            The individuals from this generation to evaluate and track fitness
            data for.
        """
        # Defer to the subclass to simulate population and evaluate fitness.
        self.evaluate_population(population)
        # Update our book keeping of individuals with the best fitness score.
        best_of_generation = max(population)
        self.best_individual_by_generation.append(best_of_generation)
        self.fitness_by_generation.append(best_of_generation.fitness)
        self.best_individual = max(best_of_generation, self.best_individual)

    def evolve(self, num_generations):
        """Evolve this lineage for the desired number of generations.

        Parameters
        ----------
        num_generations : int
            The number of iterations of the main loop of the genetic algorithm.
        """
        # Defer to the subclass to supply the initial population.
        population = self.make_initial_population()
        for generation in range(num_generations):
            self.run_generation(population)
            # Breed the next generation, unless this is the last one.
            if generation + 1 < num_generations:
                next_population = self.propagate(generation, population)
            self.per_generation_callback(generation, population)
            population = next_population

    def propagate(self, generation, population):
        """Make a new population via selection and reproduction.

        Parameters
        ----------
        generation : int
            The zero-based index of the current generation.
        population : list of Evolvable
            The candidates for breeding the next population.

        Returns
        -------
        list of Evolvable
            A new population created by pairing up individuals from the
            population parameter and calling Evolvable.breed.
        """
        # Select parents who get to breed and mates who optionally donate genes
        # to the parents for producing offspring.
        population_size = len(population)
        parents = select(population, population_size)
        mates = select(population, population_size)
        # Shuffle the mates to avoid always pairing the most fit with each
        # other and the least fit with each other.
        random.shuffle(mates)
        new_population = []
        for parent, mate in zip(parents, mates):
            # Pass in None for mate for asexual reproduction.
            if mate is parent or not parent.should_crossover():
                mate = None
            child = parent.make_offspring(mate)
            self.per_breeding_callback(generation, parent, mate, child)
            new_population.append(child)
        return new_population
