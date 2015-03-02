from numba import cuda
import numba
import numpy as np

################################################################################
# Test whether or not an integer conforms to the modified Collatz conjecture.
#
# REFERENCE HERE
#
# The Collatz conjecture.
# http://en.wikipedia.org/wiki/Collatz_conjecture
#
################################################################################
@numba.cuda.jit("void(int64[:])")
def conforms_to_conjecture(arr_a):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    i = tx + bx * bw

    if i >= arr_a.size:
        return	

    x = arr_a[i]

    loop_count = 0

    if x%3 == 0:
        return

    while not x == 1:
        if x%9 == 1:
            x = (4*x - 1)/3
        elif x%9 == 7 or x%9 == 4:
            x = (x-1)/3
        elif x%9 == 2 or x%9 == 8:
            x =(2*x-1)/3
        elif x%9 == 5:
            x = (8*x - 1) / 3

        loop_count += 1

        # This condition means that we are stuck in an effectively infinite loop
        # and that we have found an counter example.
        if loop_count>1000:
            arr_a[i] *= -1
            return

    return

################################################################################

my_gpu = numba.cuda.get_current_device()

# Starting number to test. We can change this if we are restarting after
# some number of integers that we have already tested.
counter = 0

# This the range of integers we will test.
#myrange = 500000000 #5 * 10^8
myrange = 50000000 #5 * 10^7

n = myrange  * counter # Useful for starting runs at a non-zero count

toWrite = False # Will this be running on numbers larger than we have previously tested

numbers = np.arange(myrange, dtype=np.int64)
numbers += n

thread_ct = my_gpu.WARP_SIZE
block_ct = int(myrange / thread_ct)

# Loop over an arbitrarily large number of integers to search for counter-examples
# to the modified Collatz conjecture.
while n < 10000000000000000:

    counter += 1
    
    # Call the GPU code!
    conforms_to_conjecture[block_ct, thread_ct](numbers)

    # Print the array of numbers that have failed the conjection. If this
    # array is empty, then no counterexamples have been found. 
    print np.abs(numbers[np.where(numbers<0)])

    if len(numbers[np.where(numbers<0)]>0):
        print "COUNTER EXAMPLE FOUND!!!!!!"
        exit(-1)

    # If a number that the conjecture does not hold for is found,
    # then the above method call will get stuck in an infinite loop

    print "run # ", counter, numbers[0]
    #prints the run number (aka 'counter') and the smallest
    #number the gpu is currently working with

    numbers += myrange

    if numbers[0] % 100000000000 == 0 and toWrite: #If we are on a multiple of 10^11
        #and we currently want to keep track of the number we're on
        q = numbers[0] / 100000000000   #10^11 
        f = open('currHighestNumber.txt', 'w') #Write that number / 10^11 to a file
        f.write('%d' % q)
        f.close()
        


