'''Managed memory for doing parallel computation on the GPU.

In the CUDA programming model, thousands of threads are launched simultaneously
to work on the same data objects, often with one thread responsible for a
single cell in a large multi-dimensional array. This project uses CUDA to run
many Game of Life simulations in parallel, updating every cell of several GOL
worlds simultaneously. To do this efficiently, we need to bundle the data for
all simulations into a single array, and minimize the number of times that data
gets copied to / from the GPU device.

This class is meant to solve that problem. Given the memory footprint for a
single simulation, the Arena will make an array of N copies of that footprint
for use by the GPU. It provides two views on that data: a per-simulation view
for the GameOfLifeSimulationclass to deal with the data for each instance, and
a per-data-object view which aggregates equivalent data across all simulations.
The Arena class provides basic caching so that updates to per-simulation data
get batched, avoiding many small copies to / from the GPU device.

Since this class is specifically meant to model the entire data footprint of
this program on the GPU, so exactly one instance is created automatically and
shared as global state.
'''

# TODO: implement. Instead of a singleton, just add global functions and data.
