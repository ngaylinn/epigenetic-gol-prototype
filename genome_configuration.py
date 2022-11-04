"""Classes for exploring sub-spaces of the GameOfLifeSimulation genome.

This project explores various ways to make a Genotype from a fixed genome, and
ways that Genotype will vary across generations when evolved. The GenomeConfig
class is used to represent one way that the Genome can be constrained. It is
also fits the Evolvable interface, meaning that GenomeConfigs can be evolved to
find a configuration that works well for evolving a GameOfLifeSimulation to a
particular fitness goal.
"""

import copy
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
MAX_MULTIPLIER = 2.0


class FitnessVector(IntEnum):
    """How did this individual's parent fitness compare to its parent?

    This represents the fitness trajectory of this organism's lineage, so
    mutation and crossover rates can vary depending on whether fitness is
    increasing, stagnating, or decreasing.
    """
    WORSE = 0
    SAME = 1
    BETTER = 2


class VectorizedFloat:
    """A evolvable float value that can be conditioned by a FitnessVector.

    DynamicGenomeConfig and DynamicGeneConfig evolve float values that may or
    may not be conditioned by a FitnessVector depending on their settings.
    Correctly managing these values is verbose, so this class encapsulates
    that complexity to make the rest of the code more readable.

    If constructed with use_fitness_vector set to True, this will manage a
    set of three values, one for every FitnessVector setting, that mutate
    independently. If use_fitness_vector is False, then it will instead
    construct a single value that is used for all FitnessVector settings.
    """
    def __init__(self, use_fitness_vector, max_value):
        self.use_fitness_vector = use_fitness_vector
        self.max_value = max_value
        if self.use_fitness_vector:
            self.multi_value = [self._random_value() for _ in FitnessVector]
        else:
            self.single_value = self._random_value()

    # Get a random valid setting for this value.
    def _random_value(self):
        return random.random() * self.max_value

    def vector(self):
        """Return this value's setting across all FitnessVector values."""
        if self.use_fitness_vector:
            return self.multi_value
        return [self.single_value for _ in FitnessVector]

    def get(self, fitness_vector):
        """Get the correct value for the given fitness_vector."""
        if self.use_fitness_vector:
            return self.multi_value[fitness_vector]
        return self.single_value

    def mutate(self):
        """Maybe randomly change the values in this object."""
        if self.use_fitness_vector:
            for fitness_vector in FitnessVector:
                if coin_flip(DEFAULT_MUTATION_RATE):
                    self.multi_value[fitness_vector] = self._random_value()
        elif coin_flip(DEFAULT_MUTATION_RATE):
            self.single_value = self._random_value()


class GeneConfig:
    """Metadata used for managing variation of a single Gene.

    This provides the functionality needed for the predefined GenomeConfigs.
    Gene values can either be fixed or randomized. If not fixed, they will
    mutate at the standard rate.
    """
    def __init__(self, gene, fixed_value=None):
        self.gene = gene
        self.fixed_value = fixed_value

    def get_initial_value(self):
        """Get a valid initial setting for this gene."""
        if self.fixed_value is not None:
            return self.fixed_value
        return self.gene.randomize()

    # The fitness vector is not considered here, but is used by the
    # DynamicGeneConfig subclass when overriding this method.
    def mutation_rate(self, fitness_vector, global_rate):
        """Get the mutation rate for this gene, given its context."""
        if self.fixed_value is not None:
            return 0.0
        return global_rate


class GenomeConfig:
    """Whole genome metadata for constraining possible Genotypes.

    This provides the functionality needed for the predefined GenomeConfigs.
    Each gene can be configured to have a fixed value or be randomized and
    allowed to mutate freely. Mutation and crossover rates are fixed at their
    default values.
    """
    def __init__(self, genome, fixed_values=None):
        """Constructor for GenomeConfig.

        Parameters
        ----------
        genome : dict of str : Gene
            A map from gene names to Gene objects to be managed by this
            GenomeConfig.
        fixed_values : dict of str : any
            A map from gene names to fixed values for each gene (if any).
        """
        self.genome = genome
        self.gene_configs = {}
        self._initialize_gene_configs(fixed_values or {})

    # This is separated from the __init__ method just so subclasses can
    # override it.
    def _initialize_gene_configs(self, fixed_values):
        for gene_name, gene in self.genome.items():
            fixed_value = fixed_values.get(gene_name, None)
            self.gene_configs[gene_name] = GeneConfig(gene, fixed_value)

    def initialize_genotype_data(self, data):
        """Set appropriate randomized values for all Genes in a Genotype.

        Parameters
        ----------
        data : np.ndarray
            A structured array matching the genome argument passed in on
            construction. The data in this array will be ignored and
            overwritten with appropriate randomized values.
        """
        for gene_name, gene_config in self.gene_configs.items():
            data[gene_name] = gene_config.get_initial_value()

    # fitness_vector is unused for this method, but is used by the
    # DynamicGenomeConfig subclass when overriding this method.
    def mutation_rate(self, gene_name, fitness_vector):
        """Return the mutation rate for a gene given fitness_vector."""
        return self.gene_configs[gene_name].mutation_rate(
            fitness_vector, DEFAULT_MUTATION_RATE)

    # fitness_vector is unused for this method, but is used by the
    # DynamicGenomeConfig subclass when overriding this method.
    def crossover_rate(self, fitness_vector):
        """Return the crossover rate to use given this fitness_vector."""
        return DEFAULT_CROSSOVER_RATE


class DynamicGeneConfig(GeneConfig):
    """A more complex and evolvable version of GeneConfig.

    This class allows mutation rates to vary on a gene-by-gene basis, either
    conditioned on a FitnessVector or not. It also suppots exploring random
    fixed values for each gene instead of having that pre-specified.
    """
    def __init__(self, gene, use_fitness_vector):
        super().__init__(gene)
        self.use_fitness_vector = use_fitness_vector
        # The probability 0.1 was chosen to be somewhat rare, but still common
        # enough that a few of these show up in the initial population.
        if coin_flip(0.1):
            self.fixed_value = gene.randomize()
        self.mutation_multiplier = VectorizedFloat(
            use_fitness_vector, MAX_MULTIPLIER)

    def mutation_rate(self, fitness_vector, global_rate):
        """Get the mutation rate for this gene, given its context."""
        if self.fixed_value is not None:
            return 0.0
        multiplier = self.mutation_multiplier.get(fitness_vector)
        return min(multiplier * global_rate, MAX_MUTATION_RATE)


class DynamicGenomeConfig(GenomeConfig):
    """A more complex and evolvable version of GenomeConfig.

    This class randomizes mutation and crossover rates and allows them to
    evolve. It also supports evolving per-gene mutation rates and fixed values.
    The behavior of this class is conditioned on two factors:
        - If use_fitness_vector is set to True, then all mutation and crossover
        rates will vary depending on the given FitnessVector. Otherwise, they
        will be the same regardless of the FitnessVector value.
        - If use_per_gene_config is set to True, then mutation rates and fixed
        values are allowed to evolve on a gene-by-gene basis. Otherwise, fixed
        values won't be used and mutation rates apply to all genes equally.
    """
    def __init__(self, genome, use_fitness_vector, use_per_gene_config):
        self.use_fitness_vector = use_fitness_vector
        self.use_per_gene_config = use_per_gene_config
        self.global_mutation_rate = VectorizedFloat(
            use_fitness_vector, MAX_MUTATION_RATE)
        self.global_crossover_rate = VectorizedFloat(
            use_fitness_vector, 1.0)
        super().__init__(genome, {})

    def _initialize_gene_configs(self, fixed_values):
        # If use_per_gene_config is set, then use the DynamicGeneConfig class
        # to support more complex behavior. In that case, we don't use
        # pre-specified fixed values, we let them evolve randomly.
        if self.use_per_gene_config:
            for gene_name, gene in self.genome.items():
                self.gene_configs[gene_name] = DynamicGeneConfig(
                    gene, self.use_fitness_vector)
        # Otherwise, use the static GeneConfigs setup by the parent class.
        else:
            super()._initialize_gene_configs(fixed_values)

    def __str__(self):
        """Generate a nice tabular summary of the data in this GenomeConfig.

        This method uses the same layout regardless of use_fitness_vector and
        use_per_gene_config so its easy to compare GenomeConfigs that were
        created with different settings.
        """
        row = '{0:<24} {1:<8.4f} {2:<8.4f} {3:<8.4f}'
        header = row.replace('.4f', 's')
        rows = [
            header.format('Gene', 'Worse', 'Same', 'Better'),
            row.format('global_mutation_rate',
                       *self.global_mutation_rate.vector()),
            row.format('global_crossover_rate',
                       *self.global_crossover_rate.vector()),
        ]
        for gene_name, gene_config in self.gene_configs.items():
            mutation_rates = [
                self.mutation_rate(gene_name, fitness_vector)
                for fitness_vector in FitnessVector]
            if gene_config.fixed_value is not None:
                suffix = f' Fixed value: {gene_config.fixed_value}'
            else:
                suffix = ''
            rows.append(row.format(gene_name, *mutation_rates) + suffix)
        return '\n'.join(rows)

    def mutation_rate(self, gene_name, fitness_vector):
        global_rate = self.global_mutation_rate.get(fitness_vector)
        if self.use_per_gene_config:
            return self.gene_configs[gene_name].mutation_rate(
                fitness_vector, global_rate)
        return global_rate

    def crossover_rate(self, fitness_vector):
        return self.global_crossover_rate.get(fitness_vector)


class GenomeConfigEvolvable(evolution.Evolvable):
    """An adaptor for evolving GenomeConfig objects with the evolution module.

    This class handles the work of breeding GenomeConfig objects and managing
    variation. The main purpose of this class is to clearly separate the work
    of evolving a GenomeConfig from the work of using it to evolve a
    GameOfLifeSimulation. This seems less confusing than putting all these
    similar sounding methods in a single class.

    Note that this class uses the default mutation and crossover rates. We only
    use more complicated settings when evolving a GameOfLifeSimulation.
    """
    def __init__(self, genome, use_fitness_vector, use_per_gene_config):
        super().__init__('G')
        self.genome_config = DynamicGenomeConfig(
            genome, use_fitness_vector, use_per_gene_config)
        self.use_per_gene_config = use_per_gene_config

    def should_crossover(self):
        return coin_flip(DEFAULT_CROSSOVER_RATE)

    def make_offspring(self, mate=None):
        # Start by making a deep copy of self. This way we have a valid
        # GenomeConfig object with exactly the same settings, that can be
        # changed independently with out affecting self.
        result = copy.deepcopy(self)
        child_config = result.genome_config
        # If we have a mate, then we should use it for crossover. Randomly
        # borrow setings from mate to override the ones from self.
        if mate:
            mate_config = mate.genome_config
            if coin_flip():
                child_config.global_mutation_rate = copy.deepcopy(
                    mate_config.global_mutation_rate)
            if coin_flip():
                child_config.global_crossover_rate = copy.deepcopy(
                    mate_config.global_crossover_rate)
            for gene_name in child_config.genome:
                if coin_flip():
                    child_config.gene_configs[gene_name] = copy.deepcopy(
                        mate_config.gene_configs[gene_name])
        # Mutate settings for the GenomeConfig and all its GeneConfigs.
        child_config.global_mutation_rate.mutate()
        child_config.global_crossover_rate.mutate()
        if self.use_per_gene_config:
            for gene_config in self.genome_config.gene_configs.values():
                if coin_flip(DEFAULT_MUTATION_RATE):
                    gene_config.fixed_value = gene_config.gene.randomize()
                gene_config.mutation_multiplier.mutate()
        return result
