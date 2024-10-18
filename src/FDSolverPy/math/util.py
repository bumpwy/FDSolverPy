import numpy as np
import numba as nb
import functools as ft
import math


@nb.jit(nopython=True)
def stable_sum(arr):
    curr_sum, corr = 0, 0
    for x in arr:
        t = curr_sum + x
        if np.absolute(curr_sum)>=np.absolute(x):
            corr += (curr_sum - t) + x
        else:
            corr += (x - t) + curr_sum
        # update
        curr_sum = t
    return curr_sum, corr

def stable_sum_step(a, b):
    t = a[0] + b[0]
    if np.absolute(a[0])>=np.absolute(b[0]):
        c = (a[0] - t) + b[0]
    else:
        c = (b[0] - t) + a[0]
    c += (a[1] + b[1])

    return [t,c]

if __name__ == '__main__':
    a = [1.,2.,1.+1e10,5.,10.,1-1e10]
    sum_corr = np.zeros((len(a),2))
    sum_corr[:,0] = a
    
    print('Summing the following array:',a)
    print(f'math.fsum gives: {math.fsum(a)}, while Neumaien sum gives: {stable_sum(a)}')
