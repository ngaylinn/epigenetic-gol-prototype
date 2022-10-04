"""Tools for visualizing a genetic algorithm over many generations."""

import functools

import matplotlib.pyplot as plt
import numpy as np

import gene_types
import genome


class SimulationLineageDebugger:
    """A debug tool for SimulationLineage.

    To use this class, pass a SimulationLineage to the constructor, then use
    the various break_on_* methods to indicate places where you would like to
    pause execution and inspect the population.

    Attributes
    ----------
    samples : list
        A collection of all the individuals visualized by this class. To view
        the full simulations these represent, use gol_simulation.export_videos.
    """

    def __init__(self, lineage):
        self.lineage = lineage
        self.generations_to_sample = []
        self.samples = []
        self.prev_population = None
        self.max_fitness_delta = float('inf')
        self.min_fitness_delta = float('-inf')
        self.parent_watch_list = []
        self.child_watch_list = []
        self.breeding_events = {}
        lineage.per_generation_callback = functools.partial(
            SimulationLineageDebugger._on_generation, self)
        lineage.per_breeding_callback = functools.partial(
            SimulationLineageDebugger._on_breeding, self)
        # Make sure we record the first frame of the simulation (the phenotype)
        # so we can use it in our visualizations.
        if 0 not in lineage.frames_needed:
            lineage.frames_needed = [0] + lineage.frames_needed

    def break_on_generations(self, generations):
        """Break when any of the specified generations is reached.

        Calling this method has no immediate effect, but once you call the
        lineage's evolve method, this will cause it to break at the specified
        generations and display a visualization of the full population,
        including fitness and mate selection details.

        Parameters
        ----------
        generations : list
            A list of zero-indexed generation numbers to stop at.
        """
        self.generations_to_sample.extend(generations)

    def break_on_fitness_increase(self, threshold):
        """Break when fitness increases more than threshold in one generation.

        Calling this method has no immediate effect, but once you call the
        lineage's evolve method, this will cause it to break whenever fitness
        goes up by more than threshold in a single generation. It will show a
        visualization of the population like break_on_generations for the
        generation before and after the increase happened.

        Parameters
        ----------
        threshold : int
            The minimum change in fitness score that will cause a break.
        """
        self.max_fitness_delta = threshold

    def break_on_fitness_decrease(self, threshold):
        """Break when fitness decreases more than threshold in one generation.

        Calling this method has no immediate effect, but once you call the
        lineage's evolve method, this will cause it to break whenever fitness
        goes down by more than threshold in a single generation. It will show a
        visualization of the population like break_on_generations for the
        generation before and after the decrease happened.

        Parameters
        ----------
        threshold : int
            The minimum change in fitness score that will cause a break.
        """
        self.min_fitness_delta = -threshold

    def inspect_breading(self, parent_id=None, child_id=None):
        """Visualize a breeding event once it happens.

        Calling this method has no immediate effect, but once you call the
        lineage's evolve method, this will show a visualization of the desired
        breeding events comparing the genotypes for parent, mate, and child.

        This method is meant to be used with one of the break_on_* methods. Use
        them to find the relevant simulation IDs, then add a call to this
        method to see in detail what happened between generations.

        Parameters
        ----------
        parent_id : str, optional
            Show all breeding events where an individual with this id was
            either the parent or the mate.
        child_id : str, optional
            Show all breeding events where an individual with this id was
            the child.
        """
        if parent_id:
            self.parent_watch_list.append(parent_id)
        if child_id:
            self.child_watch_list.append(child_id)

    # A callback run on each generation of the lineage, checking to see whether
    # a break is requested and visualizing the relevant data.
    def _on_generation(self, generation, population):
        prev_fitness = self.lineage.fitness_by_generation[generation - 1]
        curr_fitness = self.lineage.fitness_by_generation[generation]
        fitness_delta = curr_fitness - prev_fitness
        # If there was a big jump in fitness
        if (fitness_delta > self.max_fitness_delta or
                fitness_delta < self.min_fitness_delta):
            print(f'Breaking on generation {generation - 1} '
                  f'(fitness: {prev_fitness} -> {curr_fitness})')
            # Show the population both before and after the fitness change.
            self._visualize_population(generation - 1, self.prev_population)
            self._visualize_breeding(generation - 1)
            self._visualize_population(generation, population)
        # If this generation was specifically requested
        elif generation in self.generations_to_sample:
            print(f'Breaking on generation {generation}')
            self._visualize_population(generation, population)
            self._visualize_breeding(generation)
        # Show all the data visualizations triggered by this generation.
        plt.show()
        self.prev_population = population

    # A callback on each breeding event. We save this data rather than
    # displaying it right away, so that we can appropriately show it with the
    # relevant generation summary.
    def _on_breeding(self, generation, parent, mate, child):
        if (parent.identifier in self.parent_watch_list or
                (mate and mate.identifier in self.parent_watch_list) or
                child.identifier in self.child_watch_list):
            self.breeding_events.setdefault(generation, []).append(
                (parent, mate, child))
            for sample in [parent, mate, child]:
                if sample not in self.samples:
                    self.samples.append(sample)

    # Prints a summary of the full population in a generation. This summary
    # includes each individual's phenotype and fitness. Individuals that got to
    # reproduce are outlined in red, with a thickness proportional to the
    # number of offspring they produced.
    def _visualize_population(self, generation, population):
        self.samples.extend(population)
        fig = plt.figure(f'Generation {generation}', figsize=(16, 10))
        population = sorted(population, reverse=True)
        for index, simulation in enumerate(population):
            axis = fig.add_subplot(4, 8, index + 1)
            axis.set_title(simulation.identifier)
            axis.spines[:].set_visible(True)
            if simulation.num_children > 0:
                plt.setp(axis.spines.values(), color='#ff0000')
                plt.setp(axis.spines.values(),
                         linewidth=simulation.num_children)
            axis.set_xlabel(f'fitness: {simulation.fitness}')
            axis.tick_params(bottom=False, left=False,
                             labelbottom=False, labelleft=False)
            axis.imshow(simulation.history[0], cmap='gray', vmin=0, vmax=255)

    # Prints a side-by-side comparison of the full genotypes of parent, mate,
    # and child. Grid genes are displayed as an image, including diff images
    # that highlight just the pixels that changed. Other gene types are
    # displayed as text on the command line.
    def _visualize_breeding(self, generation):
        if generation not in self.breeding_events:
            return
        for parent, mate, child in self.breeding_events[generation]:
            mate_used = mate and child.num_parents > 1
            if mate_used:
                print(f'Breeding event: {parent.identifier} x '
                      f'{mate.identifier} -> {child.identifier}')
            else:
                print(f'Breeding event: {parent.identifier} -> '
                      f'{child.identifier}')
            for key, gene in genome.GENOME.items():
                if isinstance(gene, gene_types.GridGene):
                    fig = plt.figure(f'{child.identifier}, {key}')
                    axis = fig.add_subplot(2, 3, 1)
                    axis.set_title(parent.identifier)
                    plt.imshow(parent.genotype.data[key], cmap='gray')
                    plt.axis('off')
                    axis = fig.add_subplot(2, 3, 2)
                    axis.set_title(child.identifier)
                    plt.imshow(child.genotype.data[key], cmap='gray')
                    plt.axis('off')
                    if mate_used:
                        axis = fig.add_subplot(2, 3, 3)
                        plt.imshow(mate.genotype.data[key], cmap='gray')
                        axis.set_title(mate.identifier)
                        plt.axis('off')
                    parent_delta = np.logical_xor(
                        parent.genotype.data[key], child.genotype.data[key])
                    axis = fig.add_subplot(2, 3, 4)
                    axis.set_title(f'{parent.identifier} vs. '
                                   f'{child.identifier}')
                    plt.imshow(parent_delta, cmap='gray')
                    plt.axis('off')
                    if mate_used:
                        mate_delta = np.logical_xor(
                            mate.genotype.data[key], child.genotype.data[key])
                        axis = fig.add_subplot(2, 3, 6)
                        axis.set_title(f'{mate.identifier} vs. '
                                       f'{child.identifier}')
                        plt.imshow(mate_delta, cmap='gray')
                    plt.axis('off')
                else:
                    if mate_used:
                        print(f'{key}: {parent.genotype.data[key]} x '
                              f'{mate.genotype.data[key]} -> '
                              f'{child.genotype.data[key]}')
                    else:
                        print(f'{key}: {parent.genotype.data[key]} x '
                              f'{child.genotype.data[key]}')
