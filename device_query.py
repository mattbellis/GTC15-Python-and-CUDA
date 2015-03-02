################################################################################
# Day's second assignment : Printing information on the GPU
# 
# While not a duplicate of the CUDA deviceQuery example, this code
# uses numba.cuda to dump some minimal information about the 
# GPU device. 
#
################################################################################

import numba.cuda
import numpy as np
import math

my_gpu = numba.cuda.get_current_device()
#print my_gpu
#print type(my_gpu)

print "Running on GPU:", my_gpu.name
cores_per_capability = {
	1: 8,
	2: 32,
	3: 192,
}

cc = my_gpu.compute_capability
print "Compute capability: ", "%d.%d" %cc, 
majorcc = cc[0]
print "\nNumber of streaming pultiprocess:", my_gpu.MULTIPROCESSOR_COUNT
cores_per_multiprocessor = cores_per_capability[majorcc]
print "Number of cores per multiprocessor:", cores_per_multiprocessor
total_cores = cores_per_multiprocessor * my_gpu.MULTIPROCESSOR_COUNT
print "Number of cores on GPU:", total_cores

xDim = my_gpu.MAX_BLOCK_DIM_X
yDim = my_gpu.MAX_BLOCK_DIM_Y
zDim = my_gpu.MAX_BLOCK_DIM_Z
print "Max dimension size of a thread block (x, y, z): (", xDim, ",", yDim, ",",zDim,")"
print "Maximum number of threads per block:",  my_gpu.MAX_THREADS_PER_BLOCK
xGrid = my_gpu.MAX_GRID_DIM_X
yGrid = my_gpu.MAX_GRID_DIM_X
zGrid = my_gpu.MAX_GRID_DIM_Z
print "Max dimension size of a thread block (x, y, z): (", xGrid, ",", yGrid, ",",zGrid,")"
print "Warp Size:        ", my_gpu.WARP_SIZE
