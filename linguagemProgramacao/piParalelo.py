'''
Programa Pi Paralelo
@author: Pedro Martins Moreira Neto

Computa o valor de Pi usando N threads/processos
Nesta fase do projeto cada thread apenas imprime o valor computado. 

'''

from multiprocessing import Process
import time

def pi(start, end, step):
	print "Start: "  + str(start)
	print "End: " + str(end)
	summ = 0.0

	for i in range(start, end):
		x = (i+0.5) * step
		summ = summ + 4.0/(1.0 + x*x)

	print(summ)


if __name__ == '__main__':
	
	num_steps = 100000000 #100.000.000
	step = 1.0 / num_steps
	n = 4
	procs = []
	chunk = num_steps/n

	for i in range(n):
		procs.append(Process(target=pi, args = (i*chunk, (i+1)*chunk, step)))

	for i in range(n):
		procs[i].start()

	for i in range(n):
		procs[i].join()
