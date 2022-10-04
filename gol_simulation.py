"""Classes and functions for evolving and simulating Game of Life scenarios.

This module holds the GameOfLifeSimulation class and a few utlity methods. It
provides the interface between a population of Evolvable objects and the CUDA
kernel which runs many GOL simulations in parallel on an NVidia GPU.

An important consideration here is when to copy data from the GPU. In order to
record a video of the simulation or use frames from that video to evaluate
fitness, the frames must be copied from the GPU device to the host system.
That's an expensive operation, and naively doing it on every step of the
simulation can slow things down dramatically.

For this reason, there are two main access points for this module:
- simulate is used to launch a population of GameOfLifeSimulations on the GPU
  capturing only the frames needed for debugging and computing fitness scores.
  This is what should be used when evolving a SimulationLineage.
- record_videos is used to launch a population of GameOfLifeSimulations on the
  GPU, capturing every frame and saving the full videos to gif files. This
  should be called on small numbers of sample GameOfLifeSimulations captured
  during the evolution of a SimulationLineage.
"""

from PIL import Image

import evolution
import kernel


# The number of frames to run every Game of Life simulation.
SIMULATION_RUN_LENGTH = 100
# When exporting a simulation video, scale it up by this much to make it
# easier to see.
IMAGE_SCALE_FACTOR = 2


class GameOfLifeSimulation(evolution.Evolvable):
    """An Evolvable subclass representing a single Game of Life simulation.

    The purpose of this class is to interact with the kernel module and track
    metadata about a single Game of Life simulation, including things like
    fitness and frames from the simulation video.

    The behavior of a GameOfLifeSimulation is entirely determined by its
    Genotype, which is either provided by the developer or derived from another
    GameOfLifeSimulation's Genotype via reproduction.
    """
    def __init__(self, genotype):
        super().__init__('S')
        self.genotype = genotype
        # Frames from this organism's simulated life. This may include every
        # frame of the simulated lifetime, or just the first and / or last
        # frames. In either case, look up frames using a relative index from
        # the start or the end since the actual length may vary.
        self.frames = []
        # When this simulation is being run on the GPU, sim_index will be set
        # to indicate which of the NUM_SIMS prallel simulation slots has been
        # allocated to this GameOfLifeSimulation.
        self.sim_index = None
        # We track fitness multigenerationally, but for the first generation
        # there is no parent, so default to 0.
        self.parent_fitness = 0

    def before_run(self, sim_index):
        """Setup before running the simulation.

        Prepares to construct this organism's phenotype by pushing the
        necessary genotype data to the GPU.

        Parameters
        ----------
        sim_index : int
            The simulation index for this run.
        """
        self.sim_index = sim_index
        kernel.set_genotype(self.sim_index, self.genotype.data)
        # Clear frames before running the simulation. This way, if a
        # GameOfLifeSimulation object is reused between evolution and later
        # recording a full video, we won't mix frames from the two runs.
        self.frames = []

    def capture_frame(self):
        """Add the last computed frame to the simulation frames.
        """
        frame = kernel.get_frame(self.sim_index)
        self.frames.append(frame)

    def should_crossover(self):
        """Returns whether to perform crossover in this breeding event.

        Part of the Evolvable interface.
        """
        return self.genotype.should_crossover()

    def make_offspring(self, mate=None):
        """Construct a new organism, optionally using genes from mate.

        Part of the Evolvable interface.

        Parameters
        ----------
        mate : GameOfLifeSimulation, optional
            A second GameOfLifeSimulation whose Genotype should be used to add
            variation in the next generation. If this value is specified, that
            means that should_crossover was already called and returned True.

        Returns
        -------
        GameOfLifeSimulation
            A new GameOfLifeSimulation derived from this one (and maybe mate).
        """
        parent_fitness = self.fitness
        grandparent_fitness = self.parent_fitness
        mate_genotype = mate.genotype if mate else None
        child_genotype = self.genotype.make_offspring(
            parent_fitness, grandparent_fitness, mate_genotype)
        child = GameOfLifeSimulation(child_genotype)
        child.parent_fitness = parent_fitness
        return child

    def save_video(self, filename):
        """Export the saved frames as an animated gif file.

        Note, this will only export the frames recorded in the last simulation
        run. If you want a full video, use the record_videos function. If you
        want a partial video, specify the frames you want and call simulate.

        Parameters
        ----------
        filename : str
            The location on disk to store the gif file.
        """
        images = []
        for frame in self.frames:
            scale = IMAGE_SCALE_FACTOR
            # Scale up each frame (without interpolation, since we want to see
            # every pixel) in both the vertical and horizontal dimensions.
            resized = frame.repeat(scale, 0).repeat(scale, 1)
            images.append(Image.fromarray(resized, mode="L"))
        images[0].save(
            filename, save_all=True, append_images=images[1:], loop=0)


# Breaks up a list of GameOfLifeSimulation objects into batches that will fit
# on the GPU. This function is an iterator.
def _make_batches(population):
    batch_size = kernel.NUM_SIMS
    for index in range(0, len(population), batch_size):
        yield population[index:index + batch_size]


def simulate(population, frames_to_capture):
    """Simulate the lives of a population of GameOfLifeSimulations.

    Parameters
    ----------
    population : list of GameOfLifeSimulation
        The simulations to run on the GPU. For optimal GPU utilization, this
        list should be some multiple of kernel.NUM_SIMS in length.
    frames_to_capture : list of int
        Which frames from the simulation should be recorded. The values in this
        list should be positive or negative indexes into a list with
        SIMULATION_RUN_LENGTH values in it. For optimal performance, this list
        should be as short as possible.
    """
    # The frames_to_capture argument might have negative frame indices, so
    # normalize them to all be positive values.
    frames_to_capture = [
        index if index >= 0 else index + SIMULATION_RUN_LENGTH
        for index in frames_to_capture]
    # If we only want to record frames from the beginning, there's no point in
    # running the simulation to the end.
    steps = min(max(frames_to_capture) + 1, SIMULATION_RUN_LENGTH)
    for batch in _make_batches(population):
        # Prepare to run the simulations.
        for sim_index, simulation in enumerate(batch):
            simulation.before_run(sim_index)

        # Create the phenotype for each simulation in population to use as the
        # first frames in their respective simulations.
        kernel.make_phenotypes()
        for simulation in batch:
            if 0 in frames_to_capture:
                simulation.capture_frame()

        # Actually run the GOL simulations.
        for step in range(1, steps):
            kernel.step_simulations()
            if step in frames_to_capture:
                for simulation in batch:
                    simulation.capture_frame()


def record_videos(population, path):
    """Simulate the lives of population and save them as gif files.

    This function reevaluates the full life time of each GameOfLifeSimulation
    in population capturing each frame as it goes, which really slows things
    down. It then resizes each frame and writes data to the file system, which
    is slower still! Be careful calling this function on large populations.

    Parameters
    ----------
    population : list of GameOfLifeSimulation
        The simulations to record video of.
    path : str
        The file system path where all the videos should be saved. This path is
        assumed to exist. All gif files are saved with the simulation
        identifier (managed by Evolvable) and the suffix "_run" to distinguish
        it from any other per-simulation data that might be saved.
    """
    print(f'Recording full simulations for {len(population)}'
          ' sample individuals...')
    full_video = list(range(SIMULATION_RUN_LENGTH))
    simulate(population, frames_to_capture=full_video)
    for simulation in population:
        filename = f'{path}/{simulation.identifier}_run.gif'
        simulation.save_video(filename)
    print('All files saved.')
