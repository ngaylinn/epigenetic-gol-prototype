"""Tests for kernel.py

These test aren't meant to be especially thorough, since this is not
user-facing code. The intention is to document and provide basic sanity checks
/ regression tests for fundamental behaviors.
"""

import os.path
import unittest
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import experiments
import genome
import kernel


def make_phenotype(genotype):
    """A convenience function that returns the phenotype for genotype.

    Parameters
    ----------
    genotype : Genotype
        The genotype data used to generate the phenotype.

    Returns
    -------
    np.ndarray
        The phenotype (that is, first frame of the GameOfLifeSimulation)
        corresponding to the given genotype.
    """
    for sim_index in range(kernel.NUM_SIMS):
        kernel.set_genotype(sim_index, genotype)
    kernel.make_phenotypes()
    return kernel.get_video(0)[0]


class TestPhenotypeKernel(unittest.TestCase):
    """Validate the output of make_phenotype under a variety of configurations.

    This test generates image snapshots of the results of make_phenotype for
    reference and manual verification. There is no right or wrong here, only
    documenting what happens.
    """
    def setUp(self):
        # Seed the RNG for repeatable pseudo-random behavior.
        experiments.reset_global_state()
        # Create a simple genotype to use as a baseline for testing. Note we
        # don't bother making an actual Genotype object, since we're not
        # testing reproduction or evolution. Instead, we just mock the raw
        # Genotype data used for constructing a phenotype.
        self.base_genotype_data = np.empty((), genome.GENOME_DTYPE)
        self.base_genotype_data['seed'] = np.random.choice(
            [kernel.ALIVE, kernel.DEAD], kernel.WORLD_SHAPE)
        self.base_genotype_data['stamp'] = False
        self.base_genotype_data['stamp_offset'] = (
            kernel.WORLD_SIZE - kernel.STAMP_SIZE) / 2
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_NONE
        self.base_genotype_data['repeat_offset'] = (8, 8)
        self.base_genotype_data['mirror'] = False
        # Create a variation on the baseline genotype that sets the "stamp"
        # data to be an arrow facing up and to the left. This helps validate
        # mirroring behavior.
        self.indicator_genotype_data = self.base_genotype_data.copy()
        mid = int((kernel.WORLD_SIZE - 8) / 2)
        self.indicator_genotype_data['seed'][mid:mid+8, mid:mid+8] = [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
            [0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00],
            [0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00],
            [0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00],
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00],
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]]

    def assertGolden(self, frame):
        """Verify that frame matches output from a previous run.

        This assertion checks to see if the output from a previous run of this
        test has already been saved as a file (in which case it is presumed to
        have been reviewed and approved by a person). If it has, then the
        assertion checks that the current output matches that "golden" output
        from before. If no golden file is present, this assertion will generate
        one for the developer to review.

        Parameters
        ----------
        frame : np.ndarray of np.uint8
            An array representing a single frame from a GameOfLifeSimulation
            video. Each value in the array is a byte representing a grayscale
            value between 0 and 255.
        """
        path = 'tests/test_kernel'
        test_name = self._testMethodName
        filename = f'{path}/{self._testMethodName}.gif'
        if os.path.exists(filename):
            with Image.open(filename) as image:
                golden_frame = np.asarray(image, dtype=np.uint8) * kernel.DEAD
            if np.array_equal(frame, golden_frame):
                return
            # At this point, a golden image was found and it doesn't match the
            # argument. Display a side by side to inspect the difference.
            message = f'{test_name}: Argument does not match golden file.'
            fig = plt.figure(message)
            axis = fig.add_subplot(1, 2, 1)
            axis.set_title('argument')
            plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            axis = fig.add_subplot(1, 2, 2)
            axis.set_title('golden')
            plt.imshow(golden_frame, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.show()
            self.fail(message)
        else:
            # If the golden file wasn't found, the directory for golden files
            # might not even be set up yet, so make sure it exists.
            os.makedirs(path, exist_ok=True)
            Image.fromarray(frame, mode="L").save(filename)
            plt.figure(f'{test_name}: Please manually verify.')
            plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.show()
            self.fail('No golden file found, so the argument has been saved'
                      'as the new golden file. Please validate and delete the'
                      'file if it is not correct before rerunning this test.')

    def test_seed(self):
        """Assert basic behavior when not using a stamp."""
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_stamp(self):
        """Assert basic behavior when using a stamp."""
        self.base_genotype_data['stamp'] = True
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_stamp_with_offset(self):
        """Assert basic behavior when using a stamp."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['stamp_offset'] = (42, 27)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_once(self):
        """Stamp is used twice."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_ONCE
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_once_with_repeat_offset(self):
        """Offset between stamps works as expected."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_ONCE
        self.base_genotype_data['repeat_offset'] = (13, 21)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_once_invalid_repeat_offset(self):
        """An repeat offset of (0, 0) is the same as no repeating."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_ONCE
        self.base_genotype_data['repeat_offset'] = (0, 0)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_once_with_stamp_offset_and_repeat_offset(self):
        """Offset between stamps works as expected."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['stamp_offset'] = (42, 27)
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_ONCE
        self.base_genotype_data['repeat_offset'] = (13, 21)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_1d(self):
        """Repeat the stamp in a line."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_1D
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_1d_with_repeat_offset(self):
        """Offset between stamps in a line works as expected."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_1D
        self.base_genotype_data['repeat_offset'] = (13, 21)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_1d_with_repeat_offset_smaller_than_stamp(self):
        """Offset between stamps in a line works as expected."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_1D
        self.base_genotype_data['repeat_offset'] = (3, 5)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_2d(self):
        """Repeat the stamp in a grid pattern."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_2D
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_2d_with_repeat_offset(self):
        """Offset between stamps in a grid works as expected."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_2D
        self.base_genotype_data['repeat_offset'] = (13, 21)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_repeat_2d_with_repeat_offset_smaller_than_stamp(self):
        """Offset between stamps in a grid works as expected."""
        self.base_genotype_data['stamp'] = True
        self.base_genotype_data['repeat_mode'] = kernel.REPEAT_2D
        self.base_genotype_data['repeat_offset'] = (3, 5)
        self.assertGolden(make_phenotype(self.base_genotype_data))

    def test_mirror(self):
        """Mirroring a single stamp has no effect."""
        self.indicator_genotype_data['stamp'] = True
        self.indicator_genotype_data['mirror'] = True
        self.assertGolden(make_phenotype(self.indicator_genotype_data))

    def test_mirror_repeat_once(self):
        """When mirroring, the second stamp is mirrored relative to first."""
        self.indicator_genotype_data['stamp'] = True
        self.indicator_genotype_data['mirror'] = True
        self.indicator_genotype_data['repeat_mode'] = kernel.REPEAT_ONCE
        self.assertGolden(make_phenotype(self.indicator_genotype_data))

    def test_mirror_repeat_1d(self):
        """When mirroring, stamps alternate orientation in a line."""
        self.indicator_genotype_data['stamp'] = True
        self.indicator_genotype_data['mirror'] = True
        self.indicator_genotype_data['repeat_mode'] = kernel.REPEAT_1D
        self.assertGolden(make_phenotype(self.indicator_genotype_data))

    def test_mirror_repeat_2d(self):
        """When mirroring, stamps alternate orientation in a grid."""
        self.indicator_genotype_data['stamp'] = True
        self.indicator_genotype_data['mirror'] = True
        self.indicator_genotype_data['repeat_mode'] = kernel.REPEAT_2D
        self.assertGolden(make_phenotype(self.indicator_genotype_data))


if __name__ == '__main__':
    unittest.main()
