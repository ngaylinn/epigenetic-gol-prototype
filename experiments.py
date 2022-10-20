import functools
import math
import os
import pickle
import random
import statistics
import time

import tqdm
import numpy as np

# import debug
import evolution
import fitness
import genome
import genome_configuration
import gol_simulation
import kernel


POPULATION_SIZE = kernel.NUM_SIMS
NUM_GENOME_GENERATIONS = 100
NUM_SIMULATION_GENERATIONS = 200
NUM_TRIALS = 5

SNAPSHOT_INTERVAL = 5 * 60
PROGRESS_UPDATE_INTERVAL = 1


class ExperimentState:
    """Initializes, saves, and restores state for one experiment.

    This project typically runs studies with several experiments that may each
    run for hours at a time. For effective debugging and development its
    important to be able to stop and resume experiments, or rerun them
    repeatedly with exactly the same initial conditions.

    This class wraps a dictionary which captures the full state of an
    experiment. It manages the random number generators, Evovlable ids,
    and a progress bar UI.
    """
    def __init__(self, filename, num_simulation_generations):
        self.filename = filename
        self.num_simulation_generations = num_simulation_generations
        restored_state = self.load_from_disk()
        if restored_state:
            self.data = restored_state
            evolution.next_ids = self.data['next_ids']
            random.setstate(self.data['python_rng'])
            np.random.set_state(self.data['numpy_rng'])
        else:
            self.data = {'progress': 0}
            evolution.next_ids = {}
            random.seed(42)
            np.random.seed(42)
        self.last_save = time.time()
        self.progress_bar = None  # Initialized in the start method

    def start(self):
        """Begin running an experiment from the start or the last snapshot."""
        if self.started():
            print('Resuming from previous snapshot.')
        self.progress_bar = tqdm.tqdm(
            total=self.num_simulation_generations,
            initial=self.data['progress'],
            mininterval=1.0,
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
                try:
                    data = pickle.load(file)
                    return data
                except pickle.UnpicklingError:
                    print('Error reading pickle file {file}')

    def save_to_disk(self):
        """Save state to a pickle file on disk."""
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


class Study:
    """A set of experiments, configured with a cross product of variables.

    This project runs experiments in batches that compare performance when
    configured in different ways. The study is built from a set of variables
    and their potential values, as well as a function that actually runs the
    experiment given particular settings for those variables.

    A study is little more than a tree of dictionaries representing all the
    experiments to run. The dictionary keys represent a particular setting of
    one of the experiment variables, and the values represent all experiments
    run with that variable bound to that setting. Each of those values could be
    either another tree of experiments configured by another variable, or a
    function to call to actually run the experiment. These experiment functions
    each take a filename for the ExperimentState and return experiment
    specific data in a dictionary.
    """
    def __init__(self, variables, num_simulation_generations, run_experiment):
        """Constructor for Study

        Parameters
        ----------
        variables : list of dict of str : any
            A list of the variables under test in this study. Each item in this
            list is a dictionary representing a single variable. The values in
            this dictionary are the possible values for the given variable and
            the keys are the name associated with each variable setting.
        num_simulation_generations : int
            The number of simulation generations run for each experiment, used
            for tracking experiment progress.
        run_experiment : Callable
            A function to call to actually run the experiment. The first
            parameter to this function should be an ExperimentState object, and
            the rest should correspond to each of the variables.
        """
        self.num_simulation_generations = num_simulation_generations
        self.run_experiment = run_experiment
        self.experiment_tree = self.populate_experiment_tree(variables)

    def populate_experiment_tree(self, variables, values=()):
        """Build the dictionary tree representing all the study experiments.

        This method is called recursively in order to support an arbitrary
        number of variables.

        Paramters
        ---------
        variables : list of dict of str : any
            The variables to bind. On the initial call, this should be all the
            variables for the study.
        values : tuple of any
            The values for the variables bound so far. On the initial call,
            this should be left blank. It will be populated by recursive calls.

        Returns
        -------
        dict of str : dict or Callable
            A tree of dictionaries representing this Study. See the doc comment
            for the Study class for details.
        """
        # This wrapper handles the ExperimentState object and avoids re-running
        # the experiment if it has already completed.
        def experiment_wrapper(args, state_filename):
            state = ExperimentState(
                state_filename, self.num_simulation_generations)
            if state.finished():
                print('Reusing cached results.')
                return state.data
            return self.run_experiment(state, *args)

        experiment_tree = {}
        # The base case: we've bound all but the last of our variables.
        if len(variables) == 1:
            # For each possible value of the last variable
            for name, value in variables[0].items():
                # Bind the function used to actually run the experiment with
                # the variable values associated with this place in the
                # experiment tree. The result will be a function that takes a
                # filename for an ExperimentState object and returns the
                # results of the experiment.
                experiment_tree[name] = functools.partial(
                    experiment_wrapper, values + (value,))
        else:
            # At this point we still have several variables to bind. We need to
            # consider all permutations, so iterate through this variables
            # values and for each one expand all possible values of the
            # remaining variables.
            for name, value in variables[0].items():
                experiment_tree[name] = self.populate_experiment_tree(
                    variables[1:], values + (value,))
        return experiment_tree

    def items(self):
        """A convenience method to access the experiment tree."""
        return self.experiment_tree.items()


class SimulationLineage(evolution.Lineage):
    """A Lineage subclass for evolving GameOfLifeSimulations"""
    def __init__(self, fitness_func, genome_config):
        self.fitness_func = fitness_func
        # Note we store this as an attribute of the lineage object rather than
        # evaluating on demand so that the debugger can modify it as needed.
        self.frames_needed = fitness.get_frames_needed(fitness_func)
        self.genome_config = genome_config
        self.update_progress = lambda: None
        super().__init__()

    def make_initial_population(self):
        return [gol_simulation.GameOfLifeSimulation(
            genome.Genotype(self.genome_config))
            for _ in range(POPULATION_SIZE)]

    def evaluate_population(self, population):
        gol_simulation.simulate(
            population, frames_to_capture=self.frames_needed)
        self.update_progress()
        for simulation in population:
            simulation.fitness = self.fitness_func(simulation)


def normalize_slope(slope):
    # translate a slope value from a linear regression into a number bewteen
    # 0 and 100 indicating the angle of the line.
    return int(100 * (math.atan(slope) + math.pi / 2) / math.pi)


def median_slope(trial_series):
    median_fitness_series = list(
        map(statistics.median, zip(*trial_series)))
    generations = list(range(NUM_SIMULATION_GENERATIONS))
    slope, _ = statistics.linear_regression(generations, median_fitness_series)
    return normalize_slope(slope)


def median_slope_times_max(trial_series):
    median_fitness_series = list(
        map(statistics.median, zip(*trial_series)))
    generations = list(range(NUM_SIMULATION_GENERATIONS))
    slope, _ = statistics.linear_regression(generations, median_fitness_series)
    return normalize_slope(slope) * max(median_fitness_series)


def median_integral(trial_series):
    median_fitness_series = list(
        map(statistics.median, zip(*trial_series)))
    return sum(median_fitness_series)


def median_max_fitness(trial_series):
    median_fitness_series = list(
        map(statistics.median, zip(*trial_series)))
    return max(median_fitness_series)


GENOME_FITNESS_FUNCTIONS = {
    'median_slope_times_max': median_slope_times_max,
    'median_integral': median_integral,
    'median_max_fitness': median_max_fitness,
}


class GenomeLineage(evolution.Lineage):
    """A Lineage subclass for evolving GenomeConfigs."""
    def __init__(self, fitness_goal, genome_fitness_func):
        self.fitness_goal = fitness_goal
        self.update_progress = lambda: None
        self.genome_fitness_func = genome_fitness_func
        self.best_simulation = None
        super().__init__()

    def make_initial_population(self):
        return [
            genome_configuration.GenomeConfig(genome.GENOME)
            for _ in range(POPULATION_SIZE)
        ]

    def evaluate_population(self, population):
        for genome_config in population:
            genome_config.trial_series = []
            # There's a lot of variation in each simulation lineage, just
            # because of random chance. To try to give the GenomeLineage a
            # useful fitness signal, we run several simulated lineages and take
            # the median, as this is significantly more stable.
            for _ in range(5):
                lineage = SimulationLineage(self.fitness_goal, genome_config)
                lineage.update_progress = self.update_progress
                lineage.evolve(NUM_SIMULATION_GENERATIONS)
                # Remember trial fitness
                genome_config.trial_series.append(
                    lineage.fitness_by_generation)
                # The Lineage class will automatically keep track of the best
                # genome_config for us, but make sure we also keep track of the
                # best simulation produced by that configuration.
                genome_config.best_simulation = max(
                    self.best_simulation, lineage.best_individual)
            genome_config.fitness = self.genome_fitness_func(
                genome_config.trial_series)


def compare_phenotypes(state, fitness_func, genome_config):
    # Initialize state state, if it isn't restored from a previous snapshot.
    #state.data.setdefault('sample_simulations', [])
    state.data.setdefault('trial_series', [])
    state.data.setdefault('best_simulation', None)
    state.start()

    # Run a number of trials, either starting from the beginning or picking up
    # where we left off in the last snapshot.
    while len(state.data['trial_series']) < NUM_TRIALS:
        lineage = SimulationLineage(fitness_func, genome_config)
        lineage.update_progress = state.update_progress
        # debugger = debug.SimulationLineageDebugger(lineage)

        # Evolving a SimulationLineage only takes a few seconds, so
        # don't bother checkpointing in the midst of that process.
        lineage.evolve(NUM_SIMULATION_GENERATIONS)

        # Aggregate trial state for analysis
        state.data['trial_series'].append(lineage.fitness_by_generation)
        state.data['best_simulation'] = max((
            state.data['best_simulation'], lineage.best_individual))
        # state.data['sample_simulations'].extend(debugger.samples)

        # Consider saving a snapshot after each trial.
        state.maybe_save_snapshot()
    # Record the one best simulation across all trials.
    #state.data['sample_simulations'].append(state.data['best_simulation'])

    state.finish()
    return state.data


compare_phenotypes_study = Study(
    (fitness.EXPERIMENT_GOALS, genome.EXPERIMENT_CONFIGS),
    NUM_TRIALS * NUM_SIMULATION_GENERATIONS,
    compare_phenotypes)


def evolve_genome_config(state, fitness_goal, genome_fitness_func):
    # Initialize state state, unless we're resuming from partial data and it's
    # already initialized.
    if 'lineage' not in state.data:
        lineage = GenomeLineage(fitness_goal, genome_fitness_func)
        state.data['lineage'] = lineage
    else:
        lineage = state.data['lineage']
    # Connect the lineage with the progress bar GUI.
    lineage.update_progress = state.update_progress

    #state.data.setdefault('sample_simulations', [])
    state.start()

    # Actually evolve the GenomeConfig. After each generation, consider saving
    # a snapshot.
    while lineage.step_evolution(NUM_GENOME_GENERATIONS):
        state.maybe_save_snapshot()

    # Summarize experiment state for analysis
    best_config = lineage.best_individual
    # state.data['sample_simulations'].append(best_config.best_simulation)
    state.data['fitness_series'] = lineage.fitness_by_generation
    state.data['best_config'] = best_config
    state.data['best_trials'] = best_config.trial_series
    state.data['best_simulation'] = best_config.best_simulation

    state.finish()
    return state.data


fitness_goals = {
        fitness_name: fitness.EXPERIMENT_GOALS[fitness_name]
        for fitness_name in ['explode', 'left_to_right',
                             'symmetry', 'three_cycle']}
evolve_genome_study = Study(
    (fitness_goals, GENOME_FITNESS_FUNCTIONS),
    (NUM_GENOME_GENERATIONS * POPULATION_SIZE *
     NUM_TRIALS * NUM_SIMULATION_GENERATIONS),
    evolve_genome_config)
