"""Tests for genome.py

These test aren't meant to be especially thorough, since this is not
user-facing code. The intention is to document and provide basic sanity checks
/ regression tests for fundamental behaviors.
"""

import random
import unittest

import numpy as np

import genome
import genome_configuration
import kernel


def count_genotype_diffs(genotype_a, genotype_b, accumulator=None):
    """Compare two genotypes and report differences by gene.

    Parameters
    ----------
    genotype_a: Genotype
        The first genotype to compare
    genotype_b: Genotype
        The second genotype to compare
    accumulator: dict of string : int, optional
        A dictionary like the return value of this function. If specified, this
        method will add to the accumulator instead of starting from scratch.

    Returns
    -------
    dict of string : int
        A dictionary keyed by gene name with values indicating how many values
        differ for that gene between the two genotypes.
    """
    if accumulator is None:
        accumulator = {}
    for gene_name in genome.GENOME:
        if gene_name not in accumulator:
            accumulator[gene_name] = 0
        gene_a, gene_b = genotype_a.data[gene_name], genotype_b.data[gene_name]
        gene_diff = np.not_equal(gene_a, gene_b)
        accumulator[gene_name] += np.count_nonzero(gene_diff)
    return accumulator


class TestGenotype(unittest.TestCase):
    """Validate the behavior for creating genotypes from scratch and breeding.
    """
    def setUp(self):
        # Seed the RNGs for repeatable pseudo-random behavior. The code under
        # test uses Python and Numpy for randomness, so seed both.
        np.random.seed(42)
        random.seed(42)

    def assertProportional(self, first, second, multiple, msg=None):
        """Assert that second is within some multiple of first.

        This method does a geometric comparison of relative size difference, as
        opposed to unittest.TestCase.assertAlmostEqual, which does an
        arithmetic comparison of absolue size difference.

        Parameters
        ----------
        first : float
            The first value to compare. This value is the yardstick second is
            compared against.
        second : float
            The second value to compare.
        multiple : float
            The difference between first and second should be less than
            multiple times first.
        msg : str, optional
            The error message to show if this assertion fails.
        """
        self.assertLessEqual(abs((first - second) / first), multiple, msg)

    def test_mutations(self):
        """Per-gene mutations happen at the expected rate."""
        # The freeform GenomeConfig allows mutation for all genes.
        parent = genome.Genotype(genome.EXPERIMENT_CONFIGS['freeform'])
        gene_diffs = {}
        # This has to be pretty large since the mutation rate is pretty small.
        num_trials = 100000
        for _ in range(num_trials):
            child = parent.make_offspring(0, 0, None)
            count_genotype_diffs(parent, child, gene_diffs)
        for gene_name, gene in genome.GENOME.items():
            actual_diffs = gene_diffs[gene_name]
            num_samples = gene.num_values() * num_trials
            mutation_rate = genome_configuration.DEFAULT_MUTATION_RATE
            expected_diffs = num_samples * mutation_rate
            # Although randomized, the rate of mutations for every gene is
            # pretty close to the expected mutation rate on average.
            self.assertProportional(
                expected_diffs, actual_diffs, 0.175,
                f'{gene_name} had {actual_diffs} diffs, '
                f'expected {expected_diffs}')

    def test_crossover(self):
        """Crossover adds variety to the gene pool at the expected rate."""
        parent = genome.Genotype(genome.EXPERIMENT_CONFIGS['freeform'])
        parent.data = np.ones((), genome.GENOME_DTYPE)
        mate = genome.Genotype(genome.EXPERIMENT_CONFIGS['freeform'])
        mate.data = np.zeros((), genome.GENOME_DTYPE)
        gene_diffs = {}
        num_trials = 1000
        for _ in range(num_trials):
            if parent.should_crossover():
                child = parent.make_offspring(0, 0, mate)
            else:
                child = parent.make_offspring(0, 0, None)
            count_genotype_diffs(parent, child, gene_diffs)
        for gene_name, gene in genome.GENOME.items():
            actual_diffs = gene_diffs[gene_name]
            crossover_rate = genome_configuration.DEFAULT_CROSSOVER_RATE
            expected_diffs = (
                crossover_rate * num_trials * gene.num_values() * 0.5)
            # Although randomized, breeding produced a 50 / 50 mix of parent
            # genes at a rate pretty close to the default crossover rate.
            # Mutation is a factor here, too, but its effect is overwhelmed by
            # crossover.
            self.assertProportional(
                expected_diffs, actual_diffs, 0.1,
                f'{gene_name} had {actual_diffs} diffs,'
                f' expected {expected_diffs}')

    def test_gene_randomized(self):
        """Randomly generated genomes differ at the expected rate."""
        gene_diffs = {}
        num_trials = 100
        for _ in range(num_trials):
            # The FREEFORM GenomeConfig randomizes all genes.
            genotype_a = genome.Genotype(genome.EXPERIMENT_CONFIGS['freeform'])
            genotype_b = genome.Genotype(genome.EXPERIMENT_CONFIGS['freeform'])
            gene_diffs = count_genotype_diffs(
                genotype_a, genotype_b, gene_diffs)
        for gene_name, gene in genome.GENOME.items():
            actual_diffs = gene_diffs[gene_name]
            # When randomly picking a random gene value twice, we might pick
            # the same value by coincidence. How likely that is depends on how
            # many possible values there are to choose from.
            expected_correlation = (
                (gene.value_range() - 1) / gene.value_range())
            expected_diffs = (
                num_trials * gene.num_values() * expected_correlation)
            # Although randomized, two randomly generated genotypes correlate
            # no more than expected by chance.
            self.assertProportional(
                expected_diffs, actual_diffs, 0.1,
                f'{gene_name} had {actual_diffs} diffs, '
                f'expected {expected_diffs}')

    def test_gene_default_value_set(self):
        """Genes can be initialized to a set value instead of randomized."""
        for _ in range(10):
            # The CONTROL GenomeConfig fixes some genes.
            genotype = genome.Genotype(genome.EXPERIMENT_CONFIGS['control'])
            # Despite randomization, these genes always have the same value.
            self.assertFalse(genotype.data['stamp'])
            self.assertEqual(genotype.data['repeat_mode'], kernel.REPEAT_NONE)

    def test_gene_mutation_disabled(self):
        """Mutation can be disabled by configuration."""
        config = genome_configuration.GenomeConfig(
            genome.GENOME,
            {
                'seed': genome_configuration.GeneConfig(allow_mutation=False)
            })
        original_genotype = genome.Genotype(config)
        genotype = original_genotype
        for _ in range(10):
            genotype = genotype.make_offspring(0, 0, None)
        # Despite several rounds of reproduction, the gene with mutation
        # disallowed still has the same value.
        self.assertTrue(np.array_equal(
            original_genotype.data['seed'], genotype.data['seed']))


if __name__ == '__main__':
    unittest.main()
