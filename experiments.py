"""Functions and infrastructure to run specific genetic algorithm experiments.

This project has two different kinds of experiments: ones that evolve
GameOfLifeSimulations with a given GenomeConfig and fitness function, and ones
that evolve a GenomeConfig to maximize performance on a given fitness function.

All the experiments in this project use an ExperimentState object to track
progress and collect data. These state objects are serializable, and will
automatically save periodic snapshots to disk. If an experiment is interrupted,
it can be resumed from the last snapshot. When an experiment is completed, the
final results can be retrieved from the snapshot and used without rerunning the
experiment. To force an experiment to run from scratch, simply delete the
associated state pickle file.
"""

import os.path
import pickle
import random
import time

import tqdm
import numpy as np
import pandas as pd

# import debug
import evolution
import genome
import genome_configuration
import gol_simulation
import kernel


# The number of individuals in each generation. The same value is used for both
# SimulationLineages and GenomeLineages, although this needn't be the case.
POPULATION_SIZE = kernel.NUM_SIMS
# The number of generations to evolve a GenomeLineage for. These take a very
# long time to run, so it's fortunate they converge in fewer generations.
NUM_GENOME_GENERATIONS = 3  # 75
# The number of generations to evolve a SimulationLineage for.
NUM_SIMULATION_GENERATIONS = 3  # 200
# When evolving a GameOfLifeSimulation, run the experiment this many times to
# account for random variability.
NUM_TRIALS = 3  # 5
# The number of simulated generations in every SimulationLineage experiment.
SIMULATION_EXPERIMENT_SIZE = NUM_TRIALS * NUM_SIMULATION_GENERATIONS
# The number of simulated generations in every GenomeLineage experiment.
GENOME_EXPERIMENT_SIZE = (NUM_GENOME_GENERATIONS * POPULATION_SIZE *
                          NUM_TRIALS * NUM_SIMULATION_GENERATIONS)

# Save a snapshot of experiment state every five minutes at least.
SNAPSHOT_INTERVAL = 5 * 60
# Updating the CLI is relatively slow, so don't update the progress bar more
# often than once every second.
PROGRESS_UPDATE_INTERVAL = 1

# Variant behaviors for running a GenomeLineage. Keys are names and values are
# meant to be passed as the last two arguments to genome_experiment.
CONFIG_VARIATIONS = {
    'no_vector_x_global_only': (False, False),
    'no_vector_x_fine-grain': (False, True),
    'vector_x_global_only': (True, False),
    'vector_x_fine-grain': (True, True)
}


def reset_global_state(state=None):
    """Reset global state used by experiments to get consistent results.

    Managing global state (primarily random number generators) allows the
    experiments to produce the same results no matter what order they run in,
    or even if they got interrupted and resumed mid way.

    The ExperimentState object takes care of calling this function when running
    an experiment. It can also be called independently to manage global state
    when working directly with Genotypes and GameOfLifeSimulations without an
    experiment.

    Parameters
    ----------
    state : ExperimentState
        If provided, restore the global state from the snapshot saved here.
    """
    if state is None:
        # Reset Evolvable ids so they're unique to this experiment.
        evolution.next_ids = {}
        # Seed the RNGs at the beginning of each experiment. This way, we
        # can re-run a single experiment out of order and get the same
        # results as if we ran the whole study.
        random.seed(42)
        np.random.seed(42)
    else:
        # Each evolved individual gets a unique identifier. Remember which
        # ids were already used to avoid duplicates.
        evolution.next_ids = state.data['next_ids']
        # Restore the state of the random number generators so we get the
        # same results as if we ran the experiment straight through without
        # interruption. This project uses Python's RNG for almost
        # everything, but there are a few operations in the gene_types
        # module that get a major performance boost from using Numpy's RNG
        # instead. Manage state for both.
        random.setstate(state.data['python_rng'])
        np.random.set_state(state.data['numpy_rng'])


class ExperimentState:
    """Initializes, saves, and restores state for one experiment.

    This project runs lots of experiments, some of which may run for hours at a
    time. To avoid redundant computation and make debugging and analysis
    possible, it's important to stop and resume experiments, or rerun them
    repeatedly with exactly the same initial conditions and get exactly the
    same results.

    This class wraps a dictionary which captures the full state of an
    experiment. It manages the random number generators, Evovlable ids, and a
    progress bar UI.
    """
    def __init__(self, filename, num_simulation_generations):
        """Constructor for ExperimentState

        Parameters
        ----------
        filename : str
            The name of the file where this ExperimentState is stored.
        num_simulation_generations : int
            The number of GameOfLifeSimulation generations that will be run for
            this experiment. This is used to render a progress bar showing the
            number of generations past and remaining.
        """
        self.filename = filename
        self.num_simulation_generations = num_simulation_generations
        restored_state = self.load_from_disk()
        if restored_state:
            self.data = restored_state
            reset_global_state(self)
        else:
            self.data = {'progress': 0}
            reset_global_state()
        self.last_save = time.time()
        self.progress_bar = None  # Initialized in the start method

    def start(self, experiment_name):
        """Begin running an experiment from the start or the last snapshot."""
        if self.started():
            print(f'Resuming experiment "{experiment_name}" from snapshot...')
        else:
            print(f'Running experiment "{experiment_name}"...')
        self.progress_bar = tqdm.tqdm(
            total=self.num_simulation_generations,
            initial=self.data['progress'],
            mininterval=PROGRESS_UPDATE_INTERVAL,
            bar_format=('{n_fmt}/{total_fmt} |{bar}| '
                        'Elapsed: {elapsed} | '
                        'Remaining: {remaining}'))

    def started(self):
        """Returns True iff some experiment data has already been computed."""
        return self.data['progress'] > 0

    def update_progress(self):
        """Increment the number of simulation generations run."""
        self.progress_bar.update()
        self.data['progress'] += 1

    def finish(self):
        """Wrap up the experiment, save data, and hide the progress bar."""
        self.data['done'] = True
        self.progress_bar.close()
        self.progress_bar.clear()
        self.save_to_disk()

    def finished(self):
        """Returns True iff this experiment has already completed."""
        return 'done' in self.data

    def load_from_disk(self):
        """Attempt to load state from a pickle file on disk."""
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as file:
                data = pickle.load(file)
                return data
        return None

    def save_to_disk(self):
        """Save state to a pickle file on disk."""
        # Serialize global state before saving.
        self.data['next_ids'] = evolution.next_ids
        self.data['python_rng'] = random.getstate()
        self.data['numpy_rng'] = np.random.get_state()
        with open(self.filename, 'wb') as file:
            pickle.dump(self.data, file)
        self.last_save = time.time()

    def maybe_save_snapshot(self):
        """Save state if enough time has elapsed since the last snapshot."""
        if time.time() - self.last_save > SNAPSHOT_INTERVAL:
            self.save_to_disk()


class SimulationLineage(evolution.Lineage):
    """A Lineage subclass for evolving GameOfLifeSimulations"""
    def __init__(self, fitness_func, genome_config):
        self.fitness_func = fitness_func
        self.genome_config = genome_config
        # A customizable callback run after every simulation population. By
        # default, do nothing.
        self.update_progress_callback = lambda: None
        super().__init__()

    def make_initial_population(self):
        return [gol_simulation.GameOfLifeSimulation(
            genome.Genotype(self.genome_config))
            for _ in range(POPULATION_SIZE)]

    def evaluate_population(self, population):
        # Run all the simulations for this population.
        gol_simulation.simulate(population)
        # Evaluate fitness according to the given fitness function.
        for simulation in population:
            simulation.fitness = self.fitness_func(simulation)
        # Run the callback to track progress
        self.update_progress_callback()


def run_simulation_trials(update_progress_callback, fitness_func,
                          genome_config):
    """Run a batch of NUM_TRIALS SimulationLineages.

    Since genetic algorithms rely on randomness, results can vary significantly
    each time an lineage is run. To mitigate this, we evolve lineages in
    batches of NUM_TRIALS and average out the performance.

    Parameters
    ----------
    update_progress_callback : Callable
        A function to call after each simulated generation, to update the
        progress bar UI.
    fitness_func : Callable
        A function to evaluate the fitness of a GameOfLifeSimulation.
    genome_config : GenomeConfig
        The GenomeConfig to use when populating the SimulationLineage.

    Returns
    -------
    tuple of pd.DataFrame, GameOfLifeSimulation
        The experiment data and the best simulation found across all trials.
    """
    data_frame = pd.DataFrame(columns=['Trial', 'Generation', 'Fitness'])
    best_simulation = None
    for trial in range(NUM_TRIALS):
        # Set up the lineage and evolve it to completion.
        lineage = SimulationLineage(fitness_func, genome_config)
        lineage.update_progress_callback = update_progress_callback
        # debugger = debug.SimulationLineageDebugger(lineage)
        lineage.evolve(NUM_SIMULATION_GENERATIONS)

        # Collect data
        new_row = pd.DataFrame({
            'Trial': trial,
            'Generation': range(NUM_SIMULATION_GENERATIONS),
            'Fitness': lineage.fitness_by_generation})
        data_frame = pd.concat((data_frame, new_row))
        best_simulation = max(best_simulation, lineage.best_individual)
    return (data_frame, best_simulation)


def simulation_experiment(state_file, experiment_name,
                          fitness_func, genome_config):
    """Run an experiment to find the best simulation for a fitness goal.

    This function evolves a population of GameOfLifeSimulations created from
    the given GenomeConfig. Each simulation is run NUM_TRIALS times, since
    there's significant variation from run to run. Fitness data is recorded for
    each trial, as well as the best simulation across all trials.

    Arguments
    ---------
    state_file : str
        The name of the file used to store state for this experiment. If this
        file exists, it contains cached data from a previous run of this
        function. If the experiment completed, the data will be returned
        unmodified. If not, the experiment will continue where it left off.
        This function will save state to this location when the experiment is
        complete.
    experiment_name : str
        The name for this experiment, used to display progress in the CLI.
    fitness_func : Callable
        A function to evaluate the fitness of a GameOfLifeSimulation.
    genome_config : GenomeConfig
        The configuration used to generate the initial population and constrain
        evolution across generations.

    Returns
    -------
    ExperimentState
        The finalized data for this experiment for consumption by the
        rebuild_output module. These fields are intended for use in analysis:
        'fitness_data' : pd.DataFrame
            Data table with fitness for all generations and all trials.
        'best_simulation' : GameOfLifeSimulation
            The single simulation with highest fitness across all trials.
    """
    # Use cached results if they're available.
    state = ExperimentState(state_file, SIMULATION_EXPERIMENT_SIZE)
    if state.finished():
        return state.data

    state.start(experiment_name)
    # Running all the trials only takes a few seconds, so don't bother saving
    # snapshots in the midst of that process.
    fitness_data, best_simulation = run_simulation_trials(
        state.update_progress, fitness_func, genome_config)

    # Export data for analysis and wrap up.
    state.data['fitness_data'] = fitness_data
    state.data['best_simulation'] = best_simulation
    state.finish()
    return state.data


def weighted_median_integral(fitness_data):
    """The fitness function used to evaluate a GenomeConfig.

    When evolving a GenomeLineage, each individual is a GenomeConfig, and its
    fitness is evaluated by running a batch of NUM_TRIALS SimulationLineages
    with that configuration to see how they perform. This fitness function
    tries to capture not just the fitness of the best simulations produced in
    those trials, but also how well the genetic algorithm was able to learn and
    adapt to the fitness function.

    In short, this function looks at the area under the fitness curve, using a
    median operation to discard outlier data and giving greater weight to later
    generations. This was chosen because it has several nice properties:
        - Basing this off of an integral takes the whole process of evolution
        over generations into account rather than just the final result. It
        favors SimulationLineages whose fitness starts high and grows higher.
        - Taking the median removes single-trial outliers that most likely
        represent luck rather than a good GenomeConfig. Computing the median
        generation by generation rather than picking the median performing
        trial has the effect of smoothing results from across all trials.
        - Giving greater weight to later generations means SimulationLineages
        with increasing fitness (indicating learning) are preferred to those
        that have high fitness that doesn't improve (got stuck in a local
        maximum early on) or those that start high and then decrease (lucky
        initial population, but on average mutations just make things worse).
    """
    total = 0
    median_by_gen = fitness_data.groupby('Generation')['Fitness'].median()
    for generation, median_fitness in enumerate(median_by_gen):
        # The first generation gets a weight of 0.5 and the rest smoothly grade
        # up to a max of 1.0 for the last generation.
        weight = 0.5 + 0.5 * generation / (NUM_GENOME_GENERATIONS - 1)
        total += weight * median_fitness
    return int(total)


class GenomeLineage(evolution.Lineage):
    """A Lineage subclass for evolving GenomeConfigs.

    The individuals in this lineage are GenomeConfigs. Their fitness is
    determined by evolving a batch of NUM_TRIALS SimulationLineages and looking
    at their average performance across those trials. This is similar to
    calling simulation_experiment once for every individual in the population
    over NUM_GENOME_GENERATIONS generations.
    """
    def __init__(self, fitness_func, use_fitness_vector, use_per_gene_config):
        self.fitness_func = fitness_func
        self.use_fitness_vector = use_fitness_vector
        self.use_per_gene_config = use_per_gene_config
        self.best_simulation = None
        self.update_progress_callback = lambda: None
        super().__init__()

    def make_initial_population(self):
        return [
            genome_configuration.GenomeConfigEvolvable(
                genome.GENOME, self.use_fitness_vector,
                self.use_per_gene_config)
            for _ in range(POPULATION_SIZE)]

    def evaluate_population(self, population):
        for individual in population:
            genome_config = individual.genome_config
            fitness_data, best_simulation = run_simulation_trials(
                self.update_progress_callback,
                self.fitness_func, genome_config)
            # Attach the results from each batch of trials to the associated
            # Evolvable. This way, when we find the most fit GenomeConfig,
            # we have data about its trials.
            individual.fitness_data = fitness_data
            individual.best_simulation = best_simulation
            individual.fitness = weighted_median_integral(fitness_data)


def genome_experiment(state_file, experiment_name, fitness_func,
                      use_fitness_vector, use_per_gene_config):
    """Find the best GenomeConfig and discover what variations worked best.

    This function evolves a population of GenomeConfig objects, using each one
    to evolve a population of GameOfLifeSimulations using SimulationLineage.
    The fitness of each GenomeConfig is a function of how the SimulationLineage
    performed over multiple trials.

    This project uses three variations on how a GenomeConfig can evolve
    custom behavior. This experiment is part of a study that compares the
    impact of those three variations.

    The most basic variation is tuning global mutation and crossover rates.
    This is not uncommon for traditional genetic algorithms. Since turning this
    off would be akin to not evolving the GenomeConfig at all, it will be
    enabled for all experiments in this study as a baseline to compare the
    other two techniques against.

    A more novel variation is conditioning mutation and crossover rates on a
    FitnessVector. Theoretically, this gives the genetic algorithm a greater
    ability to steer by letting each individual in the population use feedback
    from the previous generation when adding variation for the next one. By
    turning this on and off, we can measure it's relative impact.

    Another new variation is to adjust mutation rates for each gene, including
    using a fixed value for a gene rather than allowing it to evolve freely.
    Theoretically, this gives the genetic algorithm a greater ability to focus
    its search by controlling where and how much variation is introduced from
    one generation to the next. By turning this on and off, we can measure it's
    relative impact.

    Arguments
    ---------
    state_file : str
        The name of the file used to store state for this experiment. If this
        file exists, it contains cached data from a previous run of this
        function. If the experiment completed, the data will be returned
        unmodified. If not, the experiment will continue where it left off.
        This function will save state to this location periodically and when
        the experiment is complete.
    experiment_name : str
        The name for this experiment, used to display progress in the CLI.
    fitness_func : Callable
        A function to evaluate the fitness of a GameOfLifeSimulation.
    use_fitness_vector : bool
        Whether the evolved GenomeConfigs should take the FitnessVector into
        account when computing mutation and crossover rates.
    use_per_gene_config : bool
        Whether evolved GenomeConfigs may fix values or customize mutation
        rates on a gene-by-gene basis, as opposed to just randomizing gene
        values and using the same global mutation rate for all genes.

    Returns
    -------
    ExperimentState
        The finalized data for this experiment for consumption by the
        rebuild_output module. These fields are intended for use in analysis:
        'fitness_data' : pd.DataFrame
            Data table with fitness for all generations of the GenomeLineage.
        'best_fitness_data' : pd.DataFrame
            Data table with fitness for all generations and all trials of the
            SimulationLineage for the best evolved GenomeConfig in the
            GenomeLineage.
        'best_simulation' : GameOfLifeSimulation
            The single simulation with highest fitness in this experiment.
        'best_config' : GenomeConfig
            The best performing GenomeConfig evolved in this experiment.
    """
    state = ExperimentState(state_file, GENOME_EXPERIMENT_SIZE)
    if state.finished():
        return state.data

    # Initialize a GenomeLineage, either from scratch or restoring from state.
    if 'lineage' not in state.data:
        lineage = GenomeLineage(
            fitness_func, use_fitness_vector, use_per_gene_config)
        state.data['lineage'] = lineage
    else:
        lineage = state.data['lineage']
    lineage.update_progress_callback = state.update_progress
    state.start(experiment_name)

    # Actually evolve the GenomeConfig. After each generation, consider saving
    # a snapshot.
    while lineage.step_evolution(NUM_GENOME_GENERATIONS):
        state.maybe_save_snapshot()

    # Export the fitness over generations for this experiment's GenomeLineage.
    state.data['fitness_data'] = pd.DataFrame({
        'Fitness': lineage.fitness_by_generation,
        'Generation': range(NUM_GENOME_GENERATIONS)
    })
    # Export data from the best simulation trials run by this experiment.
    state.data['best_fitness_data'] = lineage.best_individual.fitness_data
    state.data['best_simulation'] = lineage.best_individual.best_simulation
    state.data['best_config'] = lineage.best_individual.genome_config
    # Drop data that is redundant or no longer needed from state.
    del state.data['lineage']
    state.finish()
    return state.data
