import math
import multiprocessing as mp
import numpy as np
import sys
import time

def calc_pi(x):
    return 4.0 / (1 + x**2)

def np_pi(start, end, n):
    return np.sum( np.apply_along_axis( calc_pi , 0, np.linspace(start, end, n) ) )


def main(procs, steps):

	tic = time.time()
	start = list( np.linspace(0, 1, procs+1) )
	end = start[1:] + [1]

	pool = mp.Pool(processes=procs)

	result = [ pool.apply( np_pi , args=( start[i], end[i], steps/procs,) )  for i in range(procs) ]
	toc = time.time()
	print("PI: {}  Time: {}".format(sum(result)/steps, toc-tic)) 

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print("Usage: python multiprocessPiCalc-POOL.py [proccess] [steps] ")
		exit(1)
	else:
		p, s = [ int(i) for i in sys.argv[1:] ]	
	main(p, s)
