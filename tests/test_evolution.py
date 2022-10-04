"""Tests for the evolution module.

These test aren't meant to be especially thorough, since this is not
user-facing code. The intention is to document and provide basic sanity checks
/ regression tests for fundamental behaviors.
"""

import random
import unittest

import evolution


class MockEvolvable(evolution.Evolvable):
    """An instance of Evolvable with a faked fitness value."""
    def __init__(self, fitness):
        super().__init__('M')
        self.fitness = fitness

    def should_crossover(self):
        return False

    def make_offspring(self, mate=None):
        return MockEvolvable(self.fitness)


def mock_population(fitness_scores):
    """Make a population of Evolvables with the given fitness scores."""
    return [
        MockEvolvable(fitness_score) for fitness_score in fitness_scores
    ]


def count_by_fitness(population, accumulator=None):
    """Summarize the number of individuals in population by fitness.

    Parameters
    ----------
    accumulator : dict of int : int, optional
        A dictionary like the return value of this function. If specified, this
        method will add to the accumulator instead of starting from scratch.

    Returns
    -------
    dict of int : int
        A dictionary with fitness scores as keys and the total number of
        individuals in population with that fitness score as values.
    """
    if accumulator is None:
        accumulator = {}
    for individual in population:
        fitness_score = individual.fitness
        if fitness_score in accumulator:
            accumulator[fitness_score] += 1
        else:
            accumulator[fitness_score] = 1
    return accumulator


class TestSelection(unittest.TestCase):
    """Tests selection behavior used to propagate each generation."""
    def setUp(self):
        # The code under test is randomized. We want to test that pseudo-random
        # behavior, but get the same results each time we run the test.
        random.seed(42)

    def test_selection_is_proportional(self):
        """An individual is chosen proportionately to its fitness."""
        # The fittest individual accounts for half of the total fitness.
        population = mock_population([7, 1, 1, 1, 1, 1, 1, 1])
        selection = evolution.select(population, 8)
        # The fittest individual was chosen half of the time.
        self.assertEqual(count_by_fitness(selection)[7], 4)

    def test_selection_converges_on_fittest(self):
        """Over multiple generations, the population becomes more fit."""
        population = mock_population([1, 2, 3, 4, 5, 6, 7, 8])
        for _ in range(10):
            population = evolution.select(population, 8)
        # After repeatedly selecting the best from the population, the most fit
        # individuals dominate and push out the least fit ones.
        self.assertGreater(min(population).fitness, 6)

    def test_selection_is_equitable(self):
        """Over many iterations, selection is proportional to fitness."""
        fitness_scores = [1, 2, 3, 4, 5, 6, 7, 8]
        total_fitness = sum(fitness_scores)
        num_iterations = 100
        population = mock_population(fitness_scores)
        selection_counts = {}
        for _ in range(num_iterations):
            selection = evolution.select(population, len(population))
            count_by_fitness(selection, selection_counts)
        num_samples = num_iterations * len(population)
        # Although randomized and therefore inexact, over many iterations each
        # individual is selected roughly as often as expected.
        for fitness_score, sample_count in selection_counts.items():
            expected_samples = num_samples * fitness_score / total_fitness
            self.assertAlmostEqual(
                expected_samples, sample_count, delta=3)

    def test_selection_avoids_outlier_scenarios(self):
        """No individual is especially (un)lucky over many selections."""
        fitness_scores = [1, 2, 3, 4, 5, 6, 7, 8]
        total_fitness = sum(fitness_scores)
        num_iterations = 100
        population = mock_population(fitness_scores)
        for _ in range(num_iterations):
            selection = evolution.select(population, len(population))
            selection_counts = count_by_fitness(selection)
            # Despite randomization in every iteration, no individual was ever
            # selected much more or less often than its fitness score dictates.
            for fitness_score, sample_count in selection_counts.items():
                expected_samples = 8 * fitness_score / total_fitness
                self.assertAlmostEqual(expected_samples, sample_count, delta=1)


if __name__ == '__main__':
    unittest.main()
