import math

from progress.bar import IncrementalBar

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


class ProgressBar:
    """A progress bar for running GameOfLifeSimulations.

    This is a simple derivative of IncrementalBar that displays appropriate
    data for the long running simulation jobs used by this project. Since
    output to the command line interface (CLI) is a relatively slow, this class
    rate limits updates so as to not slow down the critical path of simulation.
    """
    def __init__(self, num_batches):
        batch_digits = math.floor(math.log(num_batches, 10)) + 1
        self.progress_bar = IncrementalBar(
            f'%(index){batch_digits}d/%(max)d',
            max=num_batches,
            suffix='Elapsed: %(elapsed_td)s | Remaining: %(eta_td)s')
        self.progress_bar.width = 80 - 2 * batch_digits - 42
        self.batches_so_far = 0
        self.last_update = 0

    def start(self):
        """Start the progress bar at 0 and display it on the CLI."""
        self.progress_bar.start()

    def next(self):
        """Increment progress and maybe update the display on the CLI."""
        self.batches_so_far += 1
        # Updating the CLI is relatively slow, so do we do it at most once per
        # second to mitigate the overall latency impact.
        if self.progress_bar.elapsed > self.last_update:
            self.progress_bar.goto(self.batches_so_far)
            self.last_update = self.progress_bar.elapsed

    def finish(self):
        """Complete progress and release the CLI."""
        self.progress_bar.goto(self.progress_bar.max)
        self.progress_bar.finish()


class SimulationLineage(evolution.Lineage):
    """A Lineage subclass for evolving GameOfLifeSimulations"""
    def __init__(self, fitness_func, genome_config, progress_bar):
        self.fitness_func = fitness_func
        # Note we store this as an attribute of the lineage object rather than
        # evaluating on demand so that the debugger can modify it as needed.
        self.frames_needed = fitness.get_frames_needed(fitness_func)
        self.genome_config = genome_config
        self.progress_bar = progress_bar
        super().__init__()

    def make_initial_population(self):
        return [gol_simulation.GameOfLifeSimulation(
            genome.Genotype(self.genome_config))
            for _ in range(POPULATION_SIZE)]

    def evaluate_population(self, population):
        gol_simulation.simulate(
            population, frames_to_capture=self.frames_needed)
        self.progress_bar.next()
        for simulation in population:
            simulation.fitness = self.fitness_func(simulation)


def compare_phenotypes():
    """Run the phase one experiments."""
    fitness_goals = fitness.EXPERIMENT_GOALS
    genome_configs = genome.EXPERIMENT_CONFIGS
    num_experiments = len(fitness_goals) * len(genome_configs)
    progress_bar = ProgressBar(
        num_experiments * NUM_TRIALS * NUM_SIMULATION_GENERATIONS)
    print(f'Running {len(fitness_goals)} x {len(genome_configs)} == '
          f'{num_experiments} experiments '
          f'with {NUM_TRIALS} trials for '
          f'{NUM_SIMULATION_GENERATIONS} generations each.')
    sample_simulations = []
    progress_bar.start()
    experiment_data = {}
    for fitness_name, fitness_func in fitness_goals.items():
        experiment_data[fitness_name] = {}
        for genome_name, genome_config in genome_configs.items():
            trial_data_series = []
            best_simulation = None
            for _ in range(NUM_TRIALS):
                lineage = SimulationLineage(
                    fitness_func, genome_config, progress_bar)
                # debugger = debug.SimulationLineageDebugger(lineage)
                lineage.evolve(NUM_SIMULATION_GENERATIONS)
                trial_data_series.append(lineage.fitness_by_generation)
                best_simulation = max((
                    best_simulation, lineage.best_individual))
            experiment_data[fitness_name][genome_name] = {
                'trials': trial_data_series,
                'best_id': best_simulation.identifier
            }
            # sample_simulations.extend(debugger.samples)
            sample_simulations.append(best_simulation)
    progress_bar.finish()
    print('All experiments complete.')
    return experiment_data, sample_simulations


class GenomeLineage(evolution.Lineage):
    """A Lineage subclass for evolving GenomeConfigs."""
    def __init__(self, fitness_goal, progress_bar):
        self.fitness_goal = fitness_goal
        self.progress_bar = progress_bar
        super().__init__()

    def make_initial_population(self):
        return [genome_configuration.GenomeConfig(genome.GENOME) for
                _ in range(POPULATION_SIZE)]

    def evaluate_population(self, population):
        for genome_config in population:
            trial_lineages = []
            # There's a lot of variation in each simulation lineage, just
            # because of random chance. To try to give the GenomeLineage a
            # useful fitness signal, we run several simulated lineages and take
            # the median, as this is significantly more stable.
            for _ in range(5):
                lineage = SimulationLineage(
                    self.fitness_goal, genome_config, self.progress_bar)
                lineage.evolve(NUM_SIMULATION_GENERATIONS)
                trial_lineages.append(lineage)
            trial_lineages.sort(
                key=lambda lineage: lineage.best_individual.fitness,
                reverse=True)
            lineage = trial_lineages[2]
            # TODO: Play with this. Should it be the area under the curve? The
            # rate of growth? Some other function of the lineage?
            # genome_config.fitness = sum(lineage.fitness_by_generation)
            genome_config.fitness = 0  # TODO
            # Record the lineage associated with this genome configuration so
            # when we find the most fit genome configuration we also have the
            # lineage that it produced (less fit configurations and their
            # associate lineages will be forgotten and garbage collected)
            genome_config.lineage = lineage


def evolve_genome_config():
    """Run the phase two experiments."""
    fitness_goals = ['three_cycle']
    progress_bar = ProgressBar(
        len(fitness_goals) * NUM_GENOME_GENERATIONS *
        NUM_TRIALS * POPULATION_SIZE * NUM_SIMULATION_GENERATIONS)

    sample_simulations = []
    progress_bar.start()
    experiment_data = {}
    for fitness_goal in fitness_goals:
        fitness_goal = fitness_goal.name
        genome_lineage = GenomeLineage(fitness_goal, progress_bar)
        genome_lineage.evolve(NUM_GENOME_GENERATIONS)
        experiment_data[fitness_goal] = genome_lineage.fitness_by_generation
        best_by_generation = genome_lineage.best_individual_by_generation
        for generation, genome_config in enumerate(best_by_generation):
            simulation_lineage = genome_config.lineage
            best_simulation = simulation_lineage.best_individual
            sample_simulations.append(best_simulation)
    progress_bar.finish()
    return experiment_data, sample_simulations
