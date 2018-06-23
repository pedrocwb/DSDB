import sys
import multiprocessing
from multiprocessing import Process, Pipe, current_process
import time

def worker(conn, lock, start, end, step):
    summ = 0.0

    for i in range(start, end):
        x = (i+0.5) * step
        summ = summ + 4.0/(1.0 + x*x)

    with lock:
        conn.send(summ)


def main(PROCS=4, num_steps=10000000):	
    
    lock = multiprocessing.Lock()
    step = 1.0 / num_steps
    chunk = num_steps/PROCS
    
    jobs = []
    recver, sender = Pipe()
    tic = time.time()
    for i in range(PROCS):
        p = Process(target=worker, args = (sender, lock, i*chunk, (i+1)*chunk, step))              
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

    summ = sum([recver.recv() for x in range(PROCS)]) 
    pi = summ * step

    toc = time.time()
    print("Calculated PI: {} \t time: {}".format(pi, (toc-tic)))
    


    

if __name__ == '__main__':
    args = sys.argv
    main(int(args[1]), int(args[2]))
