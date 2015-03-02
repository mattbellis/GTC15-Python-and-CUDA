from numba import cuda
import numba
import numpy as np
import math
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist
from time import time

################################################################################
# Use scipy's built in nearest neighbors function (cdist) to calculate the
# number of nearest neighbors for each point in dataset x to each point in 
# dataset y, within some radius.
################################################################################
def number_of_nearest_neighbors(x, y, radius, npts):
    X = np.vstack((x.T))
    n = np.zeros(npts)    
    top = 10000
    bottom = 0
    done = False
    # cdist chokes on too many points so we'll do 10k at a time.
    while done == False:
        ylength = len(y.T)
        #print "ylength: ",ylength

        if bottom >= ylength:
            # Normalize to the number of points.
            n /= ylength
            # Alternate normalization.
            #n /= (ylength*radius) 
            return n

        elif top >= ylength:
            top = ylength

        if(len(y) > 1):
            yPrime = y.T
            temp = np.array(yPrime[bottom:top], copy = True)
            temp = temp.T
        else:
            temp =np.array( y[bottom:top], copy = True)

        Y = np.vstack((temp.T))
        values = cdist(X, Y)
        i = 0
        for toCheck in values:
            n[i] += len(toCheck[toCheck < radius])
            i += 1

        top += 10000
        bottom += 10000

################################################################################
# 
################################################################################
@numba.cuda.jit("void(float32[:],float32[:],float32[:],float32[:],float32[:],float32)")
def number_of_nearest_neighbors_GPU(arr_ax, arr_ay, arr_bx, arr_by, arr_out, radius):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw

    # Make sure we don't calculate the thread count that is bigger than the array.
    if i>= arr_out.size:
        return

    a0x = arr_ax[i]
    a0y = arr_ay[i]

    narr_b = len(arr_bx)
    
    for j in xrange(narr_b):
        diffx = a0x - arr_bx[j]
        diffy = a0y - arr_by[j]

        distance = math.sqrt(diffx * diffx + diffy * diffy)

        # Keep track of how many points are within our radius.
        if distance<=radius:
            arr_out[i] += 1

    # Normalize to the number of points.
    arr_out[i] /= narr_b
    # Alternate normalization.
    #arr_out[i] /= (k*narr_b)



################################################################################
# Generate two randomly distributed datasets in a 2D plane.
################################################################################
npts = 10000

print "Running test for 2D datasets of %d points.\n" % (npts)

# Generate x,y points for two different datasets.
# Using random defaults to generate numbers from 0-1.
data0 = np.random.random((2,npts))
data1 = np.random.random((2,npts))

# This will be the radius in which we count points.
# So how many of data1's points will be within a circle of radius 0.1
# of each of data0's points. The value returned will be normalized to the 
# number of points in data1.
radius = 0.1

start = time()
frac_nn = number_of_nearest_neighbors(data0,data1,radius,len(data0[0]))
print "Time for scipy.cdist (CPU):   %f" % (time()-start)
#print frac_nn[0:10]


# Get the GPU info
my_gpu = numba.cuda.get_current_device()
thread_ct = my_gpu.WARP_SIZE
block_ct = int(math.ceil(float(npts) / thread_ct))

frac_nn_GPU = np.zeros(npts, dtype = np.float32)

start = time()
number_of_nearest_neighbors_GPU[block_ct, thread_ct](np.float32(data0[0]), np.float32(data0[1]), np.float32(data1[0]), np.float32(data1[1]), frac_nn_GPU, radius)
print "Time for GPU implementation:  %f" % (time()-start)

#print frac_nn_GPU[0:10]

# The sum of the difference between the two approaches should be 0, modulo
# floating point arithmetic differences between the GPU and CPU.
diff = frac_nn - frac_nn_GPU
print "\nSum of differences between two approaches: %10.20f" % np.sum(diff)

print "%d entries out of %d are not exactly 0.\n" % (len(diff[diff!=0]),npts)
print "Differences:" 
print diff[diff!=0]


