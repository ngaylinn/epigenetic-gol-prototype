"""Classes for exploring sub-spaces of the GameOfLifeSimulation genome.

This project explores various ways to make a Genotype from a fixed genome, and
ways that Genotype will vary across generations when evolved. The GenomeConfig
class is used to represent one way that the Genome can be constrained. It is
also fits the Evolvable interface, meaning that GenomeConfigs can be evolved to
find a configuration that works well for evolving a GameOfLifeSimulation to a
particular fitness goal.
"""

import random
from enum import IntEnum

import evolution
from utility import coin_flip

# Mutation and crossover rates can be evolved using GenomeConfig, but
# these traditional values are what we start with.
DEFAULT_MUTATION_RATE = 0.001
DEFAULT_CROSSOVER_RATE = 0.6

# A very high mutation rate prevents learning by evolution and also tanks
# performance by requiring too many call so the RNG. So, when evolving a
# suitable mutation rate, artificially cap it at this value.
MAX_MUTATION_RATE = 0.01


class FitnessVector(IntEnum):
    """How did this individual's parent fitness compare to its parent?

    This represents the fitness trajectory of this organism's lineage, so
    mutation and crossover rates can vary depending on whether fitness is
    increasing, stagnating, or decreasing.
    """
    WORSE = 0
    SAME = 1
    BETTER = 2


class GeneConfig:
    """Metadata used for managing variation of a single Gene.

    GenomeConfig keeps one GeneConfig object for each Gene in the genome. This
    object defines how that gene is initialized, its rate of mutation, and the
    strategy it should use for crossover. Since each gene has its own meaning,
    range of possible values, and consequences for the phenotype, it seems
    valuable to allow a different strategy for exploring the possible values
    for each gene.
    """
    def __init__(self, default_value=None, allow_mutation=True):
        """Construct a GeneConfig object.

        Parameters
        ----------
        default_value : varies by gene, optional
            If specified, use this value instead of a random value for this
            Gene when making a Genotype from scratch.
        allow_mutation : bool, optional
            If True or unspecified, then this gene is allowed to mutate between
            generations. Otherwise, this gene's value will be constant.
        """
        # TODO: Perhaps allow these to take on randomized values when evolving
        # a GenomeConfig?
        # TODO: You don't seem to be setting default_value and allow_mutation
        # separately any more, so maybe combine them?
        self.default_value = default_value
        self.allow_mutation = allow_mutation
        self.mutation_multiplier = [1.0, 1.0, 1.0]
        # TODO: Try adding back crossover strategy, but in a simpler form:
        # either remix data within a gene or don't.

    def mutation_rate(self, fitness_vector, global_rate):
        """Return the mutation rate for this gene.

        Parameters
        ----------
        fitness_vector : FitnessVector
            The fitness vector of the parent simulation.
        global_rate : float
            The global mutation rate which serves as the per-gene baseline.

        Returns
        -------
        float
            A value between 0 and MAX_MUTATION_RATE that to be used as the
            mutation rate for this gene when making an offspring from parent.
        """
        if not self.allow_mutation:
            return 0.0
        multiplier = self.mutation_multiplier[fitness_vector]
        return min(multiplier * global_rate, MAX_MUTATION_RATE)

    def mutate(self):
        """Randomly modify the attributes of this GeneConfig object.

        This method is called only when evolving a GenomeConfig.
        GeneConfig does never varies across generations of a
        SimulationLineage.
        """
        # Note that we use the DEFAULT_MUTATION_RATE here, rather than an
        # evolvable mutation rate. The buck has to stop somewhere!
        if coin_flip(DEFAULT_MUTATION_RATE):
            self.allow_mutation = not self.allow_mutation
        for index in FitnessVector:
            if coin_flip(DEFAULT_MUTATION_RATE):
                self.mutation_multiplier[index] = random.random() * 10.0

    def crossover(self, mate):
        """Randomly modify attributes to mix in values from mate.

        This method is called only when evolving a GenomeConfig. A GeneConfig
        never varies across generations of a SimulationLineage.

        Parameters
        ----------
        mate : GeneConfig
            Another GeneConfig object corresponding to the same Gene in
            a different GenomeConfig.
        """
        if coin_flip():
            self.default_value = mate.default_value
        if coin_flip():
            self.allow_mutation = mate.allow_mutation
        for index in FitnessVector:
            if coin_flip():
                mate_multipler = mate.mutation_multiplier[index]
                self.mutation_multiplier[index] = mate_multipler


class GenomeConfig(evolution.Evolvable):
    """Whole genome metadata for constraining possible Genoytpes.

    The GenomeConfig class specifies how to create Genotype objects directly
    from the genome or by breeding other Genotypes. It is responsible for
    managing crossover and mutation rates when evolving GameOfLifeSimulations.

    This project has two Evolvable classes, this one and GameOfLifeSimulation.
    This allows experiments to evolve a GenomeConfig with useful
    characteristics, then use that to evolve the best GameOfLifeSimulation
    possible using that GenomeConfig. This can be confusing, since this class
    is responsible for its own breeding and managing parts of the breeding
    process for GameOfLifeSimulations, and a lot of the same jargon comes up in
    both cases. Just remember that the breeding methods in this class are only
    used for evolving GenomeConfigs and are never used when evolving
    GameOfLifeSimulations.
    """
    def __init__(self, genome, genome_metadata=None):
        """Construct a GenomeConfig object.

        genome : dict of str : Gene
            The collection of Genes that this class is meant to configure.
        genome_metadata : dict of str : GeneConfig, optional
            A dictionary mapping from gene name to the GeneConfig object to use
            for that gene. Any gene without metadata specified will use the
            default GeneConfig behavior.
        """
        super().__init__('G')
        self.genome = genome
        if genome_metadata is None:
            genome_metadata = {}
        for gene_name in genome:
            if gene_name not in genome_metadata:
                # If no genome_metadata is specified, randomize.
                genome_metadata[gene_name] = GeneConfig()
        self.genome_metadata = genome_metadata
        self.global_mutation_rate = [DEFAULT_MUTATION_RATE] * 3
        self.global_crossover_rate = [DEFAULT_CROSSOVER_RATE] * 3

    def initialize_genotype_data(self, data):
        """Set appropriate randomized values for all Genes in a Genotype.

        Parameters
        -------
        data : np.ndarray
            A structured array matching the genome argument passed in on
            construction. The data in this array will be ignored and
            overwritten with appropriate randomized values.
        """
        for gene_name, gene in self.genome.items():
            gene_metadata = self.genome_metadata[gene_name]
            if gene_metadata.default_value is None:
                data[gene_name] = gene.randomize()
            else:
                data[gene_name] = gene_metadata.default_value

    def mutation_rate(self, gene_name, fitness_vector):
        """Return the mutation rate for a gene given fitness_vector.

        The rate of mutation can vary by gene and by this lineages historical
        fitness, and this is where that gets managed.

        Parameters
        ----------
        gene_name : str
            The name of the gene in question.
        fitness_vector : FitnessVector
            An indicator of the parent simulation's fitness trajectory used to
            determine mutaiton rates in its offspring.

        Returns
        -------
        float
            A value between 0 and MAX_MUTATION_RATE indicating the probability
            of a single-value mutation in this gene (for a gene with multiple
            values, each value should mutate independently at this rate).
        """
        # Look up the mutation rate for this gene, given the fitness_vector and
        # the global mutation rate (which also varies by fitness_vector)
        return self.genome_metadata[gene_name].mutation_rate(
            fitness_vector, self.global_mutation_rate[fitness_vector])

    def crossover_rate(self, fitness_vector):
        """Return the crossover rate to use given this fitness_vector.

        Whether to do crossover is a full Genotype consideration, so it does
        not depend on any gene.

        Parameters
        ----------
        fitness_vector : FitnessVector
            An indicator of the parent simulation's fitness trajectory used to
            determine mutaiton rates in its offspring.

        Returns
        -------
        float
            A value between 0 and 1.0 indicating the probability that this
            breeding event will use crossover (ie, sexual reproduction instead
            of asexual cloning).
        """
        return coin_flip(self.global_crossover_rate[fitness_vector])

    def should_crossover(self):
        """Return whether to crossover When making a new GenomeConfig.

        Part of the Evolvable interface. When evolving a GenomeConfig using a
        GenomeLineage, this method is used to decide whether to use crossover
        on the GenomeConfig objects. This method does NOT relate to breeding
        GameOfLifeSimulations.
        """
        # Note we use the global default rather than customizing with
        # evolution. The buck has to stop somewhere!
        return coin_flip(DEFAULT_CROSSOVER_RATE)

    def make_offspring(self, mate=None):
        """Create a new GenomeConfig by reproduction.

        Part of the Evolvable interface. When evolving a GenomeConfig using a
        GenomeLineage, this method is used to make a new GenomeConfig from a
        parent and (optionally) a mate. This method does NOT relate to breeding
        GameOfLifeSimulations.

        Parameters
        ----------
        mate : GenomeConfig, optional
            Another GenomeConfig chosen to be a mate for this breeding. If
            this value is present, that indicates that should_crossover was
            already called and returned True.
        """
        result = GenomeConfig(self.genome_metadata)
        result.global_mutation_rate = self.global_mutation_rate.copy()
        result.global_crossover_rate = self.global_crossover_rate.copy()
        if mate:
            result.crossover(mate)
        result.mutate()
        return result

    def mutate(self):
        """Modify this GenomeConfig by adding noise to its data.

        When evolving a GenomeConfig using a GenomeLineage, this method
        is used to add variation in the reproduction process.
        """
        # Mutate all global metadata in all FitnessVector conditions.
        for fitness_vector in list(FitnessVector):
            # Note we use the global default rather than customizing with
            # evolution. The buck has to stop somewhere!
            if coin_flip(DEFAULT_MUTATION_RATE):
                mutation_rate = random.random() * MAX_MUTATION_RATE
                self.global_mutation_rate[fitness_vector] = mutation_rate
            if coin_flip(DEFAULT_MUTATION_RATE):
                self.global_crossover_rate[fitness_vector] = random.random()
        # Mutate all the per-gene metadata, also.
        for gene_metadata in self.genome_metadata.values():
            gene_metadata.mutate()

    def crossover(self, mate):
        """Modify this GenomeConfig by mixing in data from mate.

        When evolving a GenomeConfig using a GenomeLineage, this method is used
        to add variation in the reproduction process.

        Parameters
        ----------
        mate : GenomeConfig
            Another GenomeConfig chosen to be a mate for this breeding.
        """
        # For all global metadata in all FitnessVector conditions, pick either
        # the self value or mate value at a 50 / 50 probability.
        for fitness_vector in list(FitnessVector):
            if coin_flip():
                mutation_rate = mate.global_mutation_rate[fitness_vector]
                self.global_mutation_rate[fitness_vector] = mutation_rate
            if coin_flip():
                crossover_rate = mate.global_crossover_rate[fitness_vector]
                self.global_crossover_rate[fitness_vector] = crossover_rate
        # Remix all the per-gene metadata, also.
        for gene_name, gene_metadata in self.genome_metadata.items():
            gene_metadata.crossover(mate.genome_metadata[gene_name])
