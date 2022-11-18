"""Classes and functions for evolving and simulating Game of Life scenarios.

This module holds the GameOfLifeSimulation class and a few utlity methods. It
provides the interface between a population of Evolvable objects and the CUDA
kernel which runs many GOL simulations in parallel on an NVidia GPU.
"""

from PIL import Image

import evolution
import kernel

# When exporting a simulation video, scale it up by this much to make it
# easier to see.
IMAGE_SCALE_FACTOR = 2
# Controls the playback speed of the animated gif, including an extended
# duration for the first frame, so you can see the phenotype clearly.
MILLISECONDS_PER_FRAME = 100
MILLISECONDS_FOR_PHENOTYPE = 10 * MILLISECONDS_PER_FRAME


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
        # Frames from this organism's simulated life. To be set by the simulate
        # function below.
        self.frames = None
        # We track fitness multigenerationally, but for the first generation
        # there is no parent, so default to 0.
        self.parent_fitness = 0

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
        assert self.frames is not None, 'Make sure you run simulate first.'
        images = []
        # Set durations for every frame. Note, this is important, since Pillow
        # will automatically drop repeated frames otherwise.
        durations = [MILLISECONDS_FOR_PHENOTYPE]
        durations.extend([MILLISECONDS_PER_FRAME] * len(self.frames))
        # Add some extra copies of the first frame to make the phenotype
        # visible.
        for frame in self.frames:
            scale = IMAGE_SCALE_FACTOR
            # Scale up each frame (without interpolation, since we want to see
            # every pixel) in both the vertical and horizontal dimensions.
            resized = frame.repeat(scale, 0).repeat(scale, 1)
            images.append(Image.fromarray(resized, mode="L"))
        images[0].save(
            filename, save_all=True, append_images=images[1:], loop=0,
            duration=durations)


# Breaks up a list of GameOfLifeSimulation objects into batches that will fit
# on the GPU. This function is an iterator.
def _make_batches(population):
    batch_size = kernel.NUM_SIMS
    for index in range(0, len(population), batch_size):
        yield population[index:index + batch_size]


def simulate(population):
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
    for batch in _make_batches(population):
        for sim_index, simulation in enumerate(batch):
            kernel.set_genotype(sim_index, simulation.genotype.data)
        kernel.make_phenotypes()
        kernel.run_simulations()
        for sim_index, simulation in enumerate(batch):
            simulation.frames = kernel.get_video(sim_index)
