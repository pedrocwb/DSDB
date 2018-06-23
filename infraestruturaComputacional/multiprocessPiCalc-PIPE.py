#
#    @Author: Pedro Martins Moreira Neto
#    @email: pedromartins.cwb@gmail.com
#
#    Cálculo do número PI com multiprocessos utilizando comunicação interprocessos através de PIPES.




from multiprocessing import Process, Pipe, current_process
import time

def worker(conn, start, end, step):
    summ = 0.0
    tic = time.time()
    for i in range(start, end):
        x = (i+0.5) * step
        summ = summ + 4.0/(1.0 + x*x)
    toc = time.time()
    print("{} range: [{} - {}] \t exec time: {}".format(current_process().name, start, end, (toc-tic)))
    conn.send(summ)
    conn.close()

def main():	
    
    n = 4
    num_steps = 1000000
    step = 1.0 / num_steps
    chunk = num_steps/n
    
    jobs = []
    pipe_list = []

    for i in range(n):
        recver, sender = Pipe()
        p = Process(target=worker, args = (sender, i*chunk, (i+1)*chunk, step))
        jobs.append(p)
        pipe_list.append(recver)
        p.start()

    for job in jobs:
        job.join()

    pi = sum([x.recv() for x in pipe_list]) / num_steps

    print("\nCalculated PI: {}".format(pi))
    


    

if __name__ == '__main__':
    main()