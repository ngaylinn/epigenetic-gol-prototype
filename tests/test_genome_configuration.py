"""Tests for genome_configuration.py

These test aren't meant to be especially thorough, since this is not
user-facing code. The intention is to document and provide basic sanity checks
/ regression tests for fundamental behaviors.
"""

import unittest

import experiments
import genome
import genome_configuration


class TestGenotype(unittest.TestCase):
    """Validate the behavior for GenomeConfigs with different settings.
    """
    def setUp(self):
        # Seed the RNGs for repeatable pseudo-random behavior.
        experiments.reset_global_state()

    def test_config_static_true(self):
        """When static, global rates use sane defaults."""
        # Check consistent behavior over 5 randomized GenomeConfigs
        configs = [
            genome_configuration.GenomeConfig(genome.GENOME, static=True)
            for _ in range(5)]
        for config in configs:
            # Each config has the default value for all mutation and crossover
            # rates.
            for fitness_vector in genome_configuration.FitnessVector:
                self.assertEqual(config.global_mutation_rate[fitness_vector],
                                 genome_configuration.DEFAULT_MUTATION_RATE)
                self.assertEqual(config.global_crossover_rate[fitness_vector],
                                 genome_configuration.DEFAULT_CROSSOVER_RATE)

    def test_static_false(self):
        """When not static, global rates vary randomly."""
        # Check consistent behavior over 5 randomized GenomeConfigs
        configs = [
            genome_configuration.GenomeConfig(genome.GENOME, static=False)
            for _ in range(5)]
        for config in configs:
            # Each config has 3 distinct values for mutation and crossover
            # rates.
            self.assertEqual(3, len(set(config.global_mutation_rate)))
            self.assertEqual(3, len(set(config.global_crossover_rate)))
            # None of those values are the defaults.
            for fitness_vector in genome_configuration.FitnessVector:
                self.assertNotEqual(
                    config.global_mutation_rate[fitness_vector],
                    genome_configuration.DEFAULT_MUTATION_RATE)
                self.assertNotEqual(
                    config.global_crossover_rate[fitness_vector],
                    genome_configuration.DEFAULT_CROSSOVER_RATE)

    def test_use_fitness_vector_true(self):
        """Mutation rates depend on FitnessVector setting."""
        # Check consistent behavior over 5 randomized GenomeConfigs
        configs = [
            genome_configuration.GenomeConfig(
                genome.GENOME, use_fitness_vector=True)
            for _ in range(5)]
        fixed_values = 0
        for config in configs:
            # Every gene has a distinct mutation rate for each setting of the
            # FitnessVector.
            for gene_name in genome.GENOME:
                mutation_rates = [
                    config.mutation_rate(gene_name, fitness_vector)
                    for fitness_vector in genome_configuration.FitnessVector]
                # If this gene doesn't have a fixed value, then it should have
                # a different mutation rate for each FitnessVector setting
                # (note I don't check for three distinct values, since the
                # MAX_MUTATION_RATE value is pretty common).
                if config.gene_configs[gene_name].fixed_value is None:
                    self.assertNotEqual(1, len(set(mutation_rates)))
                # If this gene had a fixed value, it should always have a
                # mutation rate of 0.
                else:
                    self.assertTrue(all(rate == 0 for rate in mutation_rates))
                    fixed_values += 1
        # We should expect a few genes with fixed values, but not many.
        self.assertGreater(fixed_values, 1)
        self.assertLess(fixed_values, 3)

    def test_use_fitness_vector_false(self):
        """Mutation rates DO NOT depend on FitnessVector setting."""
        # Check consistent behavior over 5 randomized GenomeConfigs
        configs = [
            genome_configuration.GenomeConfig(
                genome.GENOME, use_fitness_vector=False)
            for _ in range(5)]
        for config in configs:
            # Every gene has the same mutation rate regardless of how the
            # FitnessVector is set (or whether the gene was fixed).
            for gene_name in genome.GENOME:
                mutation_rates = [
                    config.mutation_rate(gene_name, fitness_vector)
                    for fitness_vector in genome_configuration.FitnessVector]
                self.assertEqual(1, len(set(mutation_rates)))

    def test_use_per_gene_config_true(self):
        """Mutation rate varies for each gene."""
        # Check consistent behavior over 5 randomized GenomeConfigs
        configs = [
            genome_configuration.GenomeConfig(
                genome.GENOME, use_per_gene_config=True)
            for _ in range(5)]
        fixed_values = 0
        for config in configs:
            # For every setting of the fitness vector, we see genes with
            # a variety of different mutation rates.
            for fitness_vector in genome_configuration.FitnessVector:
                mutation_rates = [
                    config.mutation_rate(gene_name, fitness_vector)
                    for gene_name in genome.GENOME]
                self.assertNotEqual(1, len(set(mutation_rates)))
            fixed_values += sum(
                config.gene_configs[gene_name].fixed_value is None
                for gene_name in genome.GENOME)
        # Some genes have randomly been assigned fixed values.
        self.assertNotEqual(0, fixed_values)

    def test_use_per_gene_config_false(self):
        """Mutation rate is the same for each gene."""
        # Check consistent behavior over 5 randomized GenomeConfigs
        configs = [
            genome_configuration.GenomeConfig(
                genome.GENOME, use_per_gene_config=False)
            for _ in range(5)]
        fixed_values = 0
        for config in configs:
            for fitness_vector in genome_configuration.FitnessVector:
                mutation_rates = [
                    config.mutation_rate(gene_name, fitness_vector)
                    for gene_name in genome.GENOME]
                self.assertEqual(1, len(set(mutation_rates)))
            fixed_values += sum(
                config.gene_configs[gene_name].fixed_value is None
                for gene_name in genome.GENOME)
        # No genes have fixed values.
        self.assertNotEqual(0, fixed_values)


if __name__ == '__main__':
    unittest.main()
