from multiprocessing import Process, Pipe


def worker(conn):
    while True:
        item = conn.recv()
        if item == 'end':
            break
        print item
    conn.send('thanks!')

def master(conn):
    conn.send(' Is')
    conn.send(' this')
    conn.send(' on?')
    conn.send('end')
    print(conn.recv())

def main():
    a, b = Pipe()
    w = Process(target = worker, args=(a, ))
    m = Process(target = master, args=(b,))

    w.start()
    m.start()
    w.join()
    m.join()

if __name__ == '__main__':
    main()