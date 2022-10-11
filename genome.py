"""Model and produce genotypes for the GameOfLifeSimulation organism.

Every GameOfLifeSimulation has its own Genotype, which species concrete values
for every gene and fully specifies the simulation's phenotype and behavior. The
GENOME, by contrast, represents the space of all possible Genotypes. It
consists of the set of Genes, their possible values, and how they may vary
across generations. In order to experiment with different behaviors, this
project also includes a GenomeConfig class, which constrains the GENOME and
reshapes the space of possible Genotypes. This project uses some predefined
GenomeConfig values as a baseline, but also uses evolution to discover more fit
GenomeConfigs.
"""


import numpy as np

import gene_types
from genome_configuration import GenomeConfig, GeneConfig, FitnessVector
import kernel
from utility import coin_flip


# The full set of genes for the GameOfLifeSimulation organism.
GENOME = {
    # A 64x64 bitmap that provides the seed data for making a phenotype. This
    # is initialized randomly, and can either be used directly or processed in
    # order to make a more complex phenotype.
    'seed': gene_types.GridGene(kernel.WORLD_SHAPE),
    # If true, crop the data in seed to an 8x8 "stamp" which will be used to
    # draw the phenotype.
    'stamp': gene_types.BoolGene(),
    # Where in the image to draw the stamp, or the origin relative to which
    # repeated copies of the stamp are drawn.
    'stamp_offset': gene_types.VectorGene(
        gene_types.IntGene(0, kernel.WORLD_SIZE - kernel.STAMP_SIZE), 2),
    # When stamp is enabled, repeat mode indicates whether and how to repeat
    # the stamp to fill the space.
    'repeat_mode': gene_types.EnumGene([
        # Don't repeat, just draw one instance of the stamp.
        kernel.REPEAT_NONE,
        # Draw exactly one additional copy of the stamp, offset from the
        # current position by repeat_offset.
        kernel.REPEAT_ONCE,
        # Draw as many copies of the stamp as will fit, laid out in a line with
        # each instance separated by repeat_offset.
        kernel.REPEAT_1D,
        # Draw as many copies of the stamp as will fit, laid out in a two
        # dimensional grid with each instance separated by repeat_offset.
        kernel.REPEAT_2D]),
    # When stamp is true and repeat_mode is not REPEAT_NONE, indicates how far
    # to move between each stamp.
    'repeat_offset': gene_types.VectorGene(
        gene_types.IntGene(0, kernel.WORLD_SIZE - kernel.STAMP_SIZE), 2),
    # When stamp is true and repeat_mode is not REPEAT_NONE, indicates that
    # every other stamp should be mirrored both horizontally and vertically.
    # This preserves structure, but allows more diverse interactions between
    # copies of the stamp.
    'mirror': gene_types.BoolGene(),
}

# Programmatically generate a struct data type representing the above GENOME.
GENOME_DTYPE = np.dtype(
    [(gene_name, gene.dtype) for gene_name, gene in GENOME.items()])


# A collection of predefined GenomeConfigs to use for these experiments, keyed
# by name.
EXPERIMENT_CONFIGS = {
    # A baseline GenomeConfig that builds the phenotype directly from a
    # 64x64 grid of bits without further elaboration.
    'control': GenomeConfig(GENOME, {
        'stamp': False}),
    'tile': GenomeConfig(GENOME, {
        'stamp': True,
        'stamp_offset': (0, 0),
        'repeat_mode': kernel.REPEAT_2D,
        'repeat_offset': kernel.STAMP_SHAPE}),
    'stamp': GenomeConfig(GENOME, {
        'stamp': True}),
    'freeform': GenomeConfig(GENOME)
}


class Genotype:
    """A concrete instantiation of GENOME for an organism.

    Properties
    ----------
    data : np.ndarray of GENOME_DTYPE
        The backing data for this Genotype, used by the kernel module to
        generate a phenotype on a GPU device.
    """
    def __init__(self, genome_config, data=None):
        """Construct a Genotype object.

        Parameters
        ----------
        genome_config : GenomeConfig
            The GenomeConfig that determines how to make this Genotype from the
            GENOME. This can be one of the pre-defined options from
            EXPERIMENT_CONFIGS above, or an evolved value.
        data : np.ndarray of GENOME_DTYPE, optional
            The backing data for this Genotype. When making a Genotype from
            scratch, this is None and data is randomly generated. When a new
            Genotype is made by reproduction, the parent passes in its data.
        """
        self.genome_config = genome_config
        if data:
            self.data = data.copy()
        else:
            self.data = np.empty((), GENOME_DTYPE)
            genome_config.initialize_genotype_data(self.data)
        # A default value for the first generation. This will be overridden by
        # make_offspring.
        self.fitness_vector = FitnessVector.SAME

    def should_crossover(self):
        """Randomly decide whether to do crossover in reproduction.

        Crossover remixes genes from two parent genotypes, hopefully preserving
        valuable traits while adding some genetic variation.
        """
        crossover_rate = self.genome_config.crossover_rate(self.fitness_vector)
        return coin_flip(crossover_rate)

    def make_offspring(self, parent_fitness, grandparent_fitness,
                       mate_genotype=None):
        """Make a new Genotype from one or two existing Genotypes.

        GameOfLifeSimulation implements the Evolvable interface and is thus
        responsible for fitness tracking and reproduction. It defers to this
        method to do the work of making a new Genotype, passing in fitness data
        as context.

        Note that fitness tracking is "matrilineal." That is, we track fitness
        for the birth parent and their birth parent and NOT the mates that
        contribute genes, for simplicity.

        Parameters
        ----------
        parent_fitness : int
            The fitness score for the parent in this breeding event. The
            Genotype of the parent GameOfLifeSimulation is the self object.
        grandparent_fitness : int
            The fitness score for the parent of the parent in this breeding
            event.
        mate_genotype : Genotype, optional
            The Genotype to combine with this one in breeding. If this value
            is present, that indicates that should_crossover was already called
            and returned True.
        """
        child = Genotype(self.genome_config, self.data)
        if parent_fitness > grandparent_fitness:
            child.fitness_vector = FitnessVector.BETTER
        elif parent_fitness < grandparent_fitness:
            child.fitness_vector = FitnessVector.WORSE
        else:
            child.fitness_vector = FitnessVector.SAME
        if mate_genotype:
            child.crossover(mate_genotype)
        child.mutate()
        return child

    def crossover(self, mate_genotype):
        """Perform crossover, randomly adopting genes from mate_genotype."""
        for gene_name, gene in GENOME.items():
            self.data[gene_name] = gene.crossover(
                self.data[gene_name], mate_genotype.data[gene_name])

    def mutate(self):
        """Maybe modify parts of this genotype at random."""
        for gene_name, gene in GENOME.items():
            mutation_rate = self.genome_config.mutation_rate(
                gene_name, self.fitness_vector)
            gene_data = self.data[gene_name]
            self.data[gene_name] = gene.mutate(gene_data, mutation_rate)
