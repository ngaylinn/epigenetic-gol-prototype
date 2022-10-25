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

# GeneConfig determines the mutation rate for a gene by applying a multipler to
# the global mutation rate. The value of this multiplier ranges from 0 to
# MAX_MULTIPLIER, inclusive.
MAX_MULTIPLIER = 10.0


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
    def __init__(self, gene, fixed_value=None, static=False):
        """Construct a GeneConfig object.

        Parameters
        ----------
        fixed_value : varies by gene, optional
            If specified, treat this gene as being fixed to the given value. It
            will not be randomized or allowed to mutate.
        """
        self.gene = gene
        if static:
            self.fixed_value = fixed_value
            self.mutation_multiplier = [1.0, 1.0, 1.0]
        else:
            # At this point, we're randomizing the GeneConfig settings, but we
            # still want to respect fixed_value if it was specified.
            self.fixed_value = fixed_value
            # Don't use fixed values for really big compound genes, since
            # fixing values mostly only makes sense for configuration genes.
            if fixed_value is None and gene.num_values() <= 2:
                # If a fixed value wans't specified, maybe set one at random.
                # The probability 0.1 was chosen to be somewhat rare, but still
                # common enough that a few of these show up in the initial
                # population.
                self.fixed_value = gene.randomize() if coin_flip(0.1) else None
            self.mutation_multiplier = []
            for _ in FitnessVector:
                multiplier = random.random() * MAX_MULTIPLIER
                self.mutation_multiplier.append(multiplier)

    def copy(self):
        """Make a copy of a GeneConfig, with its own copy of the data."""
        # It's safe to pass fixed_value by reference, since it is never
        # modified in place. It's safe to leave out the static parameter to the
        # constructor because this method is only called when static is False
        # (that is, for breeding new GenomeConfigs)
        result = GeneConfig(self.gene, self.fixed_value)
        # Make a copy of the mutation_multiplier list, so mutations in the
        # child won't modify the data in the parent.
        result.mutation_multiplier = self.mutation_multiplier.copy()
        return result

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
        if self.fixed_value is not None:
            return 0.0
        multiplier = self.mutation_multiplier[fitness_vector]
        return min(multiplier * global_rate, MAX_MUTATION_RATE)

    def get_initial_value(self):
        """Return an appropriate value to initialize this gene in the genotype.

        Returns
        -------
        varies by gene
            Either the fixed value for this gene specified in the constructor,
            or a randomized value provided by the relevent Gene subclass.
        """
        if self.fixed_value is not None:
            return self.fixed_value
        return self.gene.randomize()

    def mutate(self):
        """Randomly modify the attributes of this GeneConfig object.

        This method is called only when evolving a GenomeConfig.
        GeneConfig does never varies across generations of a
        SimulationLineage.
        """
        # Note that we use the DEFAULT_MUTATION_RATE here, rather than an
        # evolvable mutation rate. The buck has to stop somewhere!
        if coin_flip(DEFAULT_MUTATION_RATE):
            self.fixed_value = self.gene.randomize()
        for index in FitnessVector:
            if coin_flip(DEFAULT_MUTATION_RATE):
                multiplier = random.random() * MAX_MULTIPLIER
                self.mutation_multiplier[index] = multiplier

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
            self.fixed_value = mate.fixed_value
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
    def __init__(self, genome, fixed_values=None, static=False,
                 use_fitness_vector=True, use_per_gene_config=True):
        """Construct a GenomeConfig object.

        genome : dict of str : Gene
            The collection of Genes that this class is meant to configure.
        fixed_values : dict of str : varies by gene
            A dictionary mapping from gene name to a fixed value to use for
            that gene. If a fixed value is not specified for a gene, it will be
            randomized instead.
        static : bool
            True iff this GenomeConfig will not be evolved, indicating that
            attributes should be set to sensible defaults rather. Otherwise,
            randomize attributes to make a diverse starting ppopulation that
            can be tuned by evolution.
        use_fitness_vector : bool
            Whether or not to condition mutation and crossover rates on parent
            fitness. For the sake of consistency, this doesn't effect how a
            GenomeConfig is constructed or evolved, just what information is
            taken into account when computing mutation and crossover rates.
        use_per_gene_config : bool
            Whether or not to allow fixed values and mutation rates for each
            gene. For the sake of consistency, this doesn't effect how a
            GenomeConfig is constructed or evolved, just how genotypes get
            initialized and how mutation rates are computed.
        """
        super().__init__('G')
        self.genome = genome
        self.use_fitness_vector = use_fitness_vector
        self.use_per_gene_config = use_per_gene_config
        self.gene_configs = {}
        # Create a GeneConfig object for every gene in the genome, using a
        # fixed value if specified, and otherwise allowing the gene to take on
        # random values that vary across generations.
        for gene_name, gene in genome.items():
            fixed_value = None
            if fixed_values and gene_name in fixed_values:
                fixed_value = fixed_values[gene_name]
            self.gene_configs[gene_name] = GeneConfig(
                gene, fixed_value, static)
        if static:
            self.global_mutation_rate = [DEFAULT_MUTATION_RATE] * 3
            self.global_crossover_rate = [DEFAULT_CROSSOVER_RATE] * 3
        else:
            self.global_mutation_rate = []
            self.global_crossover_rate = []
            for _ in FitnessVector:
                mutation_rate = random.random() * MAX_MUTATION_RATE
                self.global_mutation_rate.append(mutation_rate)
                self.global_crossover_rate.append(random.random())

    @classmethod
    def make_static(cls, genome, fixed_values=None):
        """Make a GenomeConfig with predetermined values that can't evolve.

        This is used for the predefined GenomeConfigs defined in the genome
        module. A GenomeConfig won't evolve outside of a GenomeLineage, but its
        initialization is different depending on whether its intended for that
        purpose or not. Setting static=True means the GenomeConfig will use
        sensible defaults for values for some things, instead of randomizing
        them so task-specific values can be evolved.
        """
        return cls(genome, fixed_values=fixed_values, static=True)

    def __str__(self):
        """Generate a nice tabular summary of the data in this GenomeConfig."""
        row = '{0:<24} {1:<8.4f} {2:<8.4f} {3:<8.4f}'
        header = row.replace('.4f', 's')
        rows = [
            header.format('Gene', 'Worse', 'Same', 'Better'),
            row.format('global_mutation_rate', *self.global_mutation_rate),
            row.format('global_crossover_rate', *self.global_crossover_rate),
        ]
        if not self.use_fitness_vector:
            rows.insert(0, 'FitnessVector locked to SAME')
        if not self.use_per_gene_config:
            rows.insert(0, 'Per-gene settings IGNORED')
        for gene_name, gene_config in self.gene_configs.items():
            mutation_rates = [
                gene_config.mutation_rate(
                    vector, self.global_mutation_rate[vector])
                for vector in FitnessVector]
            if gene_config.fixed_value is not None:
                suffix = f' Fixed value: {gene_config.fixed_value}'
            else:
                suffix = ''
            rows.append(row.format(gene_name, *mutation_rates) + suffix)
        return '\n'.join(rows)

    def initialize_genotype_data(self, data):
        """Set appropriate randomized values for all Genes in a Genotype.

        Parameters
        -------
        data : np.ndarray
            A structured array matching the genome argument passed in on
            construction. The data in this array will be ignored and
            overwritten with appropriate randomized values.
        """
        # If we're using per-gene configurations, let the GeneConfig decide
        # what the initial value should be (fixed values allowed)
        if self.use_per_gene_config:
            for gene_name, gene_config in self.gene_configs.items():
                data[gene_name] = gene_config.get_initial_value()
        # Otherwise, randomize the initial value for all genes.
        else:
            for gene_name, gene in self.genome.items():
                data[gene_name] = gene.randomize()

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
        # If we're not using the fitness_vector, then ignore that information.
        # Just use the same fitness_vector value every time (other values are
        # still evolved, but are never used and have no effect on fitness)
        if not self.use_fitness_vector:
            fitness_vector = FitnessVector.SAME
        # If using per-gene configuration, let the GeneConfig determine the
        # mutation rate given the global rate and the fitness_vector.
        if self.use_per_gene_config:
            return self.gene_configs[gene_name].mutation_rate(
                fitness_vector, self.global_mutation_rate[fitness_vector])
        # Otherwise, just use the same mutation rate for all genes.
        return self.global_mutation_rate[fitness_vector]

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
        # If we're not using the fitness_vector, then ignore that information.
        # Just use the same fitness_vector value every time (other values are
        # still evolved, but are never used and have no effect on fitness)
        if not self.use_fitness_vector:
            fitness_vector = FitnessVector.SAME
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
        result = GenomeConfig(self.genome)
        result.gene_configs = {
            gene_name: gene_config.copy()
            for gene_name, gene_config in self.gene_configs.items()
        }
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
        for gene_config in self.gene_configs.values():
            gene_config.mutate()

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
        for gene_name, gene_config in self.gene_configs.items():
            gene_config.crossover(mate.gene_configs[gene_name])
