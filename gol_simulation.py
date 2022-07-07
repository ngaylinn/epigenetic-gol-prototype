'''A GA organism class representing a single Game of Life simulation.
'''
from functools import total_ordering
import kernel
import reproduction


FRAME_HISTORY_SIZE = 4


@total_ordering
class GameOfLifeSimulation:
    '''A class for a GA organism representing a single Game of Life simulation.
    '''
    def __init__(self, genotype, experiment_config):
        self.genotype = genotype
        self.experiment_config = experiment_config
        self.identifier = genotype.identifier
        self.history = []
        self.last_step = None
        # To be set by the simulator.
        self.sim_index = None
        self.fitness = None

    def before_run(self):
        '''Setup before running the simulation.
        '''
        # Actually, we should just build this in the parallel make_phenotype.
        # self.arena.push(self.sim_index, 'last_frame', self.phenotype)
        pass

    def before_step(self, logger):
        '''Setup before running a single simulation step.
        '''
        if logger.record_video:
            # TODO: Get the actual last frame, once that exists.
            last_frame = self.genotype.genes[kernel.RAW_SEED]
            logger.log_frame(self.identifier, last_frame)

    def after_step(self, step):
        '''Tear down after running a single simulation step.
        '''
        # TODO: Get the actual last frame, once that exists.
        last_frame = self.genotype.genes[kernel.RAW_SEED]

        # If the simulation is just starting or it has already ended, then
        # don't bother figuring out whether it should stop.
        if len(self.history) < FRAME_HISTORY_SIZE or self.last_step:
            return

        # If the last frame we computed is identical to a previous frame, then
        # the simulation has settled into a steady state. Record that.
        for frame in self.history:
            if frame == last_frame:
                self.last_step = step

        # Update the history with the new frame.
        self.history.append(last_frame)
        self.history = self.history[-FRAME_HISTORY_SIZE:]

    def after_run(self):
        '''Tear down after running the simulation
        '''
        pass

    def make_genotype(self, mate, logger):
        '''Produce a genotype for this organism's offspring.
        '''
        return reproduction.make_genotype(
            self.experiment_config.reproduction_config, self, mate, logger)

    def __eq__(self, other):
        return other is not None and self.fitness == other.fitness

    def __lt__(self, other):
        return other is not None and self.fitness < other.fitness
