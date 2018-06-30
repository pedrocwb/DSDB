import sys
import multiprocessing
from multiprocessing import Process, Pipe, current_process, Value
import time


def worker(pi, start, end, step):
    summ = 0.0

    for i in range(start, end):
        x = (i+0.5) * step
        summ = summ + 4.0/(1.0 + x*x)

    with pi.get_lock():
        pi.value += summ


def main(PROCS=4, num_steps=10000000):	
    
    pi = Value('d', 0.0)
    step = 1.0 / num_steps
    chunk = num_steps/PROCS
    
    jobs = []

    tic = time.time()
    for i in range(PROCS):
        p = Process(target=worker, args = (pi, i*chunk, (i+1)*chunk, step))              
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

    pi.value *= step

    toc = time.time()
    print("Calculated PI: {} \t time: {}".format(pi.value, (toc-tic)))
    



main()
