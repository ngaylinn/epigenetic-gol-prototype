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

import abc
import functools
import random

import pandas as pd


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
    sample_points = [sample_offset + i * sample_period for i in range(count)]

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
    for sample in sample_points:
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


next_ids = {}


@functools.total_ordering  # Sortable by fitness.
class Evolvable(abc.ABC):
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
        id_number = next_ids.setdefault(id_prefix, 0)
        next_ids[id_prefix] = id_number + 1
        self.identifier = f'{id_prefix}{id_number:08d}'
        # To be set by subclasses
        self.fitness = None

    @abc.abstractmethod
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

    @abc.abstractmethod
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
        return (other is not None and
                isinstance(other, self.__class__) and
                self.fitness == other.fitness)

    def __lt__(self, other):
        return (other is not None and
                isinstance(other, self.__class__) and
                self.fitness < other.fitness)


class Lineage(abc.ABC):
    """Evolve an experimental population and capture statistics.

    This class handles the basic book keeping and statistics tracking of a
    genetic algorithm. To use it, make a subclass and implement
    make_initial_population and evaluate_population for your needs.

    Attributes
    ----------
    fitness_data : pd.DataFrame
        A Pandas DataFrame with the fitness data for all organisms across all
        generations.
    best_individual : Evolvable
        The single most fit individual found across all generations.
    """
    def __init__(self):
        self.fitness_data = pd.DataFrame()
        self.generation = 0
        self.population = None
        self.best_individual = None
        # Hooks for the debugger to override. By default, they do nothing.
        self.inspect_generation_callback = lambda generation, population: None
        self.inspect_breeding_callback = lambda gen, parent, mate, child: None

    def __getstate__(self):
        # Filter out any callback functions when pickling a Lineage or any
        # Lineage subclass. Conceptually, callbacks aren't serializable,
        # they're part of how serializable data objects connect with a running
        # process. Practically speaking, pickle chokes on locally defined
        # lambdas as well as the progress bar library used in the experiments
        # module, so filtering them out is necessary.
        return {key: value
                for key, value in self.__dict__.items()
                if 'callback' not in key}

    def __setstate__(self, state_dict):
        self.__dict__ = state_dict
        # Restore dummy callbacks so we can call them without checking for
        # their presence.
        self.inspect_generation_callback = lambda generation, population: None
        self.inspect_breeding_callback = lambda gen, parent, mate, child: None

    @abc.abstractmethod
    def make_initial_population(self):
        """Return a list of Evolvables to serve as the seed population.

        Returns
        -------
        list of Evolvable
            All the individuals for the first generation.
        """
        raise NotImplementedError

    @abc.abstractmethod
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

        # Log fitness data for all simulations.
        generation_data = pd.DataFrame({
            'Generation': self.generation,
            'Identifier': [individual.identifier for individual in population],
            'Fitness': [individual.fitness for individual in population]})
        self.fitness_data = pd.concat((self.fitness_data, generation_data))

        # Keep track of the best individual across all generations.
        self.best_individual = max(max(population), self.best_individual)

    def step_evolution(self, num_generations):
        """Run the evolution of this lineage forward by one step.

        This method exists in order to support the pause / resume functionality
        in the experiments module. It makes it possible to interrupt evolution
        after each cycle, and to pick back up where we left off. This method
        tracks the current generation and population on this Lineage object so
        that the full state of this process can be serialized and deserialized
        with the Lineage object.

        Parameters
        ----------
        num_generations : int
            The number of iterations to evolve this lineage for.

        Returns
        -------
        bool
            True if evolution should continue after calling this function,
            False if the last generation has already been reached.
        """
        if self.generation == 0:
            # Defer to the subclass to supply the initial population.
            self.population = self.make_initial_population()
        if self.generation >= num_generations:
            return False
        self.run_generation(self.population)
        # Breed the next generation, unless this is the last one.
        if self.generation + 1 < num_generations:
            next_population = self.propagate(self.generation, self.population)
        # Call the callback with the current population, but only after
        # computing the next population. This way, the debugger can be aware of
        # any children the current population produced.
        self.inspect_generation_callback(self.generation, self.population)
        self.generation += 1
        if self.generation < num_generations:
            self.population = next_population
            return True
        return False

    def evolve(self, num_generations):
        """Evolve this lineage for the desired number of generations.

        Parameters
        ----------
        num_generations : int
            The number of iterations to evolve this lineage for.
        """
        self.generation = 0
        while self.step_evolution(num_generations):
            pass

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
            self.inspect_breeding_callback(generation, parent, mate, child)
            new_population.append(child)
        return new_population
