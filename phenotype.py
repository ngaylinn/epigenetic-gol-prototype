
CLEAR = 0b00000000
SET =   0b00001111
KEEP =  0b11110000


def and_part(mask_cell):
    return mask_cell & 0b11110000

def or_part(mask_cell):
    return mask_cell & 0b00001111


WORLD_SIZE = 64
SEED_MASK_SIZE = 8
NUM_TRANSFORMS = 4


def make_phenotype(raw_seeds, seed_masks, transformers, phenotypes):
    row, col, sim_id = cuda.grid(3)
    if not in_bounds(row, col):
        return

    raw_seed = raw_seeds[sim_id]
    seed_mask = seed_masks[sim_id]
    transformers = transformers[sim_id]
    phenotype = phenotypes[sim_id]

    mask_row = int(row / (WORLD_SIZE / SEED_MASK_SIZE))
    mask_col = int(col / (WORLD_SIZE / SEED_MASK_SIZE))
    mask_cell = seed_mask[mask_row][mask_col]
    and_mask = and_part(mask_cell)
    or_mask = or_part(mask_cell)
    phenotype[row][col] = (raw_seed[row][col] & and_mask) | or_mask

    for transformer in transformers:
        matcher = transformer[0]
        replacement_value = transformer[1]
        # TODO convolve

def make_all_phenotypes(arena):
    arena.update_device_data()
    # Lookup the arguments to the kernel call.
    raw_seeds = arena.multi_ref('raw_seed')
    seed_masks = arena.multi_ref('seed_mask')
    phenotype = arena.multi_ref('simulation_frame')
