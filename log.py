'''Capture metrics and videos of the GA in action.
'''
import csv
from PIL import Image

FRAME_HISTORY_SIZE = 4


def _export_csv(log_data, filename):
    with open(filename, 'w', newline='', encoding='ASCII') as file:
        writer = csv.writer(file)
        for row in log_data:
            writer.writerow(row)


def _export_gif(frames, filename):
    images = [Image.fromarray(frame, mode="L") for frame in frames]
    images[0].save(filename, save_all=True, append_images=images[1:])


class Logger:
    '''An class to collect log events and export them to the filesystem.

    The general model here is to construct a Logger object for a batch of
    experiments where you want to compare results within that batch. The Logger
    object will be passed around to any code that needs to log events. Calling
    a log* method records an event and may update the Logger's state to reflect
    that. Calling an export* method will dump the requested log object(s) to
    the filesystem.
    '''
    def __init__(self):
        self.experiment_config = None
        self.record_video = False
        self.current_generation = 0
        self.stats = [(
            'fitness_config',
            'reproduction_config',
            'phenotype_config',
            'generation',
            'best_organism',
            'fitness',
            'elapsed_time',
        )]
        self.geneology = [(
            'fitness_config',
            'reproduction_config',
            'phenotype_config',
            'generation',
            'offspring',
            'parent',
            'mate',
            'did_crossover',
        )]
        self.genotype_videos = {}
        self.simulation_videos = {}

    def export_genotype_video(self, organism_id, filename):
        '''Export an animated gif of this organism's phenotype creation.
        '''
        if organism_id in self.genotype_videos:
            _export_gif(self.genotype_videos[organism_id], filename)

    def export_simulation_video(self, organism_id, filename):
        '''Export an animated gif of this organism's life.
        '''
        if organism_id in self.simulation_videos:
            _export_gif(self.simulation_videos[organism_id], filename)

    def export_stats(self, filename):
        '''Export a CSV file of all evolution stats recorded by this logger.
        '''
        _export_csv(self.stats, filename)

    def export_geneology(self, filename):
        '''Export a CSV file of all geneology data recorded by this logger.
        '''
        _export_csv(self.geneology, filename)

    def log_experiment(self, experiment_config):
        '''Log the beginning of a new experiment.
        '''
        self.experiment_config = experiment_config
        self.current_generation = 0
        # For efficiency, we don't save videos across multiple experiments. So
        # make sure to call export_simulation_video before the next generation.
        self.simulation_videos.clear()

    def log_generation(self, best_organism, elapsed_time):
        '''Log the completion of one generation of evolution.
        '''
        self.stats.append((
            self.experiment_config.fitness_config.name,
            self.experiment_config.reproduction_config.name,
            self.experiment_config.phenotype_config.name,
            self.current_generation,
            best_organism.identifier,
            best_organism.fitness,
            elapsed_time,
        ))
        self.current_generation += 1

    def log_breeding(self, organism_id, parent_id, mate_id, did_crossover):
        '''Log the creation of one new genotype.
        '''
        self.geneology.append([
            self.experiment_config.fitness_config.name,
            self.experiment_config.reproduction_config.name,
            self.experiment_config.phenotype_config.name,
            self.current_generation,
            organism_id,
            parent_id,
            mate_id,
            did_crossover,
        ])

    def log_frame(self, organism_id, frame):
        '''Log one frame of an organism's simulated life to export as video.
        '''
        if self.record_video:
            video = self.simulation_videos.get(organism_id, [])
            video.append(frame)
            self.simulation_videos[organism_id] = video
