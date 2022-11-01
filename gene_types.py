"""Classes representing individual genes in a genome.

The main purpose of this project is to make a genetic algorithm that has an
explicit model of the genome so the evolutionary process can influence the
interpretation and variation of the genes in that genome. To do this, we use a
collection of classes to model genes of different sorts and their basic
operations.

Each gene is responsible for initializing genotype data (either randomly or
using a fixed value), as well as handling the mutation and crossover
operations, which are designed to be efficient and semantically relevant for
the given data type.

Traditional genetic algorithms typically use a string or homogenous list to
represent the genotype and don't have complex or mixed gene types. This code
could be implemented as a layer of interpretation over such a simple genotype,
but this code is more readable and more efficient than doing it that way.
"""

import abc
import random

import numpy as np

import kernel
from utility import coin_flip


class Gene(abc.ABC):
    """An abstract base class for a single gene in a genome.

    The Gene class represents the possibe values of a gene and how those values
    vary across generations. It does not hold a concrete value for a gene, but
    is used by the Genotype class to generate concrete gene values that get
    passed into the GameOfLifeSimulation class. See documentation in the genome
    module for more about the relationship between Genes and Genotypes.

    The methods of Gene subclasses usually take concrete gene values as
    parameters and / or return them as results.
    """
    def __init__(self, dtype):
        # The data type for this gene, used to allocate memory when
        # interfacing with kernel.make_phenotypes.
        self.dtype = np.dtype(dtype)

    @abc.abstractmethod
    def value_range(self):
        """The range of values this gene can accept.

        This is used for testing, in order to calculate how likely it is for
        two random gene values to be coincidentally the same.
        """
        raise NotImplementedError

    def num_values(self):
        """The number of values in this gene (one, except for arrays).

        This is used for testing, in order to calculate the number of coin flip
        events in mutation / crossover.
        """
        return self.dtype.itemsize

    @abc.abstractmethod
    def randomize(self):
        """Return a randomized valid setting for this gene."""
        raise NotImplementedError

    @abc.abstractmethod
    def mutate(self, data, mutation_rate):
        """Return a copy of data, maybe with randomized mutations.

        Parameters
        ----------
        data : varies by subclass
            The backing data for one instance of this gene. It should be of the
            type specified in self.dtype.
        mutation_rate : float
            A number between 0 and 1 representing how likely a mutation is for
            each value in data.

        Returns
        -------
        varies by subclass
            A copy of data, possibly with values changed from mutation.
        """
        raise NotImplementedError

    def crossover(self, parent_data, mate_data):
        """Return a copy of parent_data, maybe with mate_data randomly mixed in.

        Override this method if your Gene subclass has multiple values
        internally that can be recombined during crossover.

        Parameters
        ----------
        parent_data : varies by subclass
            The backing data for the instance of this gene owned by the parent
            that will produce offspring. It should be of the type specified in
            self.dtype.
        mate_data : varies by subclass
            The backing data for the instance of this gene owned by the mate
            that donates genes for this breeding. It should be of the type
            specified in self.dtype.

        Returns
        -------
        varies by subclass
            New backing data for this gene derived from parent_data and
            mate_data.
        """
        return parent_data if coin_flip() else mate_data


# For performance reasons, this isn't a generic ArrayGene that can hold values
# of any type like VectorGene. This is used for the 64x64 seed gene, and it's
# about 10x faster to use Numpy's RNG to produce 64x64 arrays of noise than to
# call the Python RNG 4096 times. This code gets used a lot, so the time saved
# by not deferring to a sub_gene is substantial.
class GridGene(Gene):
    """A gene representing a monochrome bitmap as a 2D byte array."""
    def __init__(self, shape):
        super().__init__((np.uint8, shape))
        self.shape = shape

    def value_range(self):
        return 2

    def randomize(self):
        # Fill the grid with random noise, but randomly vary how many pixels
        # get set for increased variation.
        density = random.random()
        random_floats = np.random.random_sample(self.shape)
        return kernel.DEAD * np.array(random_floats > density, dtype=np.uint8)

    def mutate(self, data, mutation_rate):
        rows, cols = self.shape
        # Usually we're only flipping one or two bits out of 4096 when this
        # method gets called. Numpy provides a handy function to sample a
        # binomial distribution, which lets us decide how many mutations to
        # make in one go. This is much faster than generating 4096 random
        # bools, using either Numpy's or Python's RNG.
        num_mutations = np.random.binomial(rows * cols, mutation_rate)
        for _ in range(num_mutations):
            row = random.randrange(rows)
            col = random.randrange(cols)
            data[row][col] ^= 0xff
        return data

    def crossover(self, parent_data, mate_data):
        # Randomly choose whether to cut the data in half vertically or
        # horizontally.
        axis = int(coin_flip())
        split_parent = np.split(parent_data, 2, axis)
        split_mate = np.split(mate_data, 2, axis)
        # Randomly merge half of one gene with half of the other.
        if coin_flip():
            return np.concatenate((split_parent[0], split_mate[1]), axis)
        return np.concatenate((split_mate[0], split_parent[1]), axis)


class EnumGene(Gene):
    """A gene representing one of several possible values."""
    def __init__(self, values):
        super().__init__(np.uint8)
        self.values = values

    def value_range(self):
        return len(self.values)

    def randomize(self):
        return random.choice(self.values)

    def mutate(self, data, mutation_rate):
        # Mutations are restricted to adjacent values, which are assumed to be
        # more similar to each other in function. This is just meant to make
        # these mutations less disruptive.
        if coin_flip(mutation_rate):
            index = self.values.index(data)
            if index == 0:
                index += 1
            elif index + 1 == len(self.values):
                index -= 1
            else:
                index = random.choice((index + 1, index - 1))
            data = self.values[index]
        return data


class BoolGene(Gene):
    """A gene representing a boolean value."""
    def __init__(self):
        super().__init__(bool)

    def value_range(self):
        return 2

    def randomize(self):
        return coin_flip()

    def mutate(self, data, mutation_rate):
        if coin_flip(mutation_rate):
            data = not data
        return data


class IntGene(Gene):
    """A gene representing an integer between min_value and max_value."""
    def __init__(self, min_value, max_value):
        super().__init__(np.uint8)
        self.min_value = min_value
        self.max_value = max_value

    def value_range(self):
        return self.max_value - self.min_value

    def randomize(self):
        return random.randint(self.min_value, self.max_value)

    def mutate(self, data, mutation_rate):
        if coin_flip(mutation_rate):
            data = self.randomize()
        return data


class VectorGene(Gene):
    """A gene representing a 1D array of genes of any type.

    This is mostly just a convenience for grouping related genes together, for
    instance turning two integers into a coordinate pair.
    """
    def __init__(self, sub_gene, size):
        super().__init__((sub_gene.dtype, size))
        self.sub_gene = sub_gene
        self.size = size

    def value_range(self):
        return self.sub_gene.value_range()

    def randomize(self):
        return tuple(self.sub_gene.randomize() for _ in range(self.size))

    def mutate(self, data, mutation_rate):
        # Note that randomly deciding whether to perform a mutation is handled
        # by the sub_gene, not the vector.
        for index in range(self.size):
            data[index] = self.sub_gene.mutate(data[index], mutation_rate)
        return data

    def crossover(self, parent_data, mate_data):
        for index in range(self.size):
            if coin_flip():
                parent_data[index] = mate_data[index]
        return parent_data
