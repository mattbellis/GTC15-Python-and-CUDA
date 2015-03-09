# FIRST ASSIGNMENT
# Use the GPU to add the elements of two arrays together and save them in a third array

from numba import cuda 
import numba
import numpy as np
import math
from time import time

my_gpu = numba.cuda.get_current_device()
@numba.cuda.jit("void(float32[:],float32[:],float32[:])")
def vadd(arr_a,arr_b,arr_out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x  #Use these data values to calculate the specific index
    i = tx + bx * bw      #You want this particular thread to work with
    if i>= arr_out.size:  #If the index is greater than the array size, there is no arithmetic to be done
        return            #This is possible due to taking the ceiling of the size of the
    arr_out[i] = arr_a[i]+arr_b[i] #Arrays divided by the thread count as the block count


n = 10000000  #How big are the arrays?
a = np.arange(n,dtype=np.float32)
b = np.arange(n,dtype=np.float32)  #Set both input arrays equal to ascending integer values from 0 to n-1
vector_sum_gpu = np.empty_like(a)  #Create a target array with the same number of elements as our input arrays
vector_sum_python = np.empty_like(a)  #Create a target array with the same number of elements as our input arrays

print "Running comparison with %d-sized arrays\n" % (n)

# Add the two vectors using pure python
'''
start = time()
for i in xrange(n):
    vector_sum_python[i] = a[i] + b[i]
print "Time to run using python:     %f" % (time()-start)
'''

# Add the two vectors using just numpy
start = time()
vector_sum = a + b
print "Time to run using numpy:      %f" % (time()-start)


#Set the thread count to the number of threads on our GPU
thread_ct = my_gpu.WARP_SIZE
#Set the block count
block_ct = int(math.ceil(float(n) / thread_ct))

#Call vadd
start = time()
vadd[block_ct, thread_ct](a, b, vector_sum_gpu)
print "Time to run using numba cuda: %f" % (time()-start)

diff = vector_sum-vector_sum_gpu

print "\nDifference of two approaches"
print diff

print "\nSum of difference of two approaches"
print diff.sum()

