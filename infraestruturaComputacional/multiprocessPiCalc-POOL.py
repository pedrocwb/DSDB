from multiprocessing import Pool
import numpy as np
import time, itertools


def pi(x): 
	return 4.0/(1.0 + x*2)

def nppi(st, end, n):
	return np.sum(np.apply_along_axis(pi, 0, np.linspace(st, end, n)))

num_steps = 10000000
n = 4	
bnds = np.linspace(0, 1, n+1)



pool = Pool(processes = n)
results = [pool.apply(pi, step)]

print(sum(results) / num_steps)



	
