from multiprocessing import Process, Queue ,current_process
import time, itertools

PI = Queue()

def pi(start, end, step):
	summ = 0.0
	tic = time.time()
	for i in range(start, end):
		x = (i+0.5) * step
		summ = summ + 4.0/(1.0 + x*x)
	toc = time.time()
	PI.put(summ)
	print("{} range: [{} - {}] \t exec time: {}".format(current_process().name, start, end, (toc-tic)))	


if __name__ == '__main__':
	
	num_steps = 100000000 #100.000.000
	step = 1.0 / num_steps
	n = 6
	procs = []
	chunk = num_steps/n

	for i in range(n):
		procs.append(Process(target=pi, args = (i*chunk, (i+1)*chunk, step)))

	for i in range(n):
		procs[i].start()

	summ = 0.0
	for i in range(n):
		procs[i].join()
		summ += PI.get()

	print("\nCalculated PI: {}".format(summ/num_steps))

	
