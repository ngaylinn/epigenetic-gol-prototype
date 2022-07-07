'''Configuration objects for GA experiments.

The purpose of this project is to compare the performance of different
strategies of reproduction and phenotype generation against a variety of
fitness functions. This means we'll be running the same scenarios again and
again with different combinations of settings. This module manages all the
possible settings.

To use, construct an ExperimentConfig object for each call to multiverse.evolve
that will be passed around to all the code that needs it.
'''

from dataclasses import dataclass
from enum import Enum


class ReproductionConfig(Enum):
    '''Experiment configuration for genotype creation and breeding.
    '''
    # Simple global mutation and crossover only.
    CONTROL = 1
    # Condition mutation by whether fitness is improving or not.
    VECTOR = 2
    # Customize mutation and crossover per gene.
    GRANULAR = 3
    # Use a fitness vector to perform granular mutation and crossover.
    BOTH = 4


class PhenotypeConfig(Enum):
    '''Experiment configuration for generating phenotypes.
    '''
    # Use just a raw seed (64x64 monochrome bitmap) for the phenotype.
    CONTROL = 1
    # Overlay a low-frequency mask above the raw seed so the phenotype can
    # quickly erase large portions of the world.
    MASK = 2
    # Do pattern matching find and replace over the seed.
    TRANSFORMERS = 3
    # Use both a mask and transformers.
    BOTH = 4


class FitnessConfig(Enum):
    '''Select which fitness function organisms will be judged by.
    '''
    # The simulation should settle in about 100 steps.
    RUN_LENGTH = 1


@dataclass
class ExperimentConfig:
    '''A collection of all configurations for a GA run.
    '''
    reproduction_config: ReproductionConfig
    phenotype_config: PhenotypeConfig
    fitness_config: FitnessConfig
