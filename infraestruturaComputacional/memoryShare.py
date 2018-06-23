from multiprocessing import Process, current_process, Queue
import itertools


ITEMS = Queue()

for i in [1, 2, 3, 4, 5, 6, 'end', 'end', 'end']:
    ITEMS.put(i)


def worker(items):
    for i in itertools.count():
        item = items.get()
        if item == 'end':
            break

    print current_process().name, "processed %i items ." % i 

if __name__ == "__main__":
    workers = [ Process( target=worker, args= (ITEMS, )) for i in range(6) ]
    for worker in workers :
        worker.start() 
    for worker in workers :
        worker.join()
    
    print "ITEMS after all workers finished : ", ITEMS

