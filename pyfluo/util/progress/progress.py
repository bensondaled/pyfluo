import time, sys
import multiprocessing as mp

class Progress(mp.Process):
    def __init__(self, verbose=True):
        super(Progress, self).__init__()
        self.daemon = True

        self.t0 = time.time()
        self.on = mp.Value('i', 1)
        self.complete = mp.Value('i', 0)
        self.verbose = verbose

    def run(self):
        while self.on.value and self.verbose:
            elap = time.time() - self.t0
            sys.stdout.write("\r%i seconds elapsed..."%elap)
            sys.stdout.flush()
            time.sleep(0.001)
        sys.stdout.write('\n')
        sys.stdout.flush()
        self.complete.value = 1

    def __enter__(self):
        self.start()
    def __exit__(self, typ, value, traceback):
        self.on.value = False
        while not self.complete.value:
            pass
        self.terminate()

if __name__ == '__main__':
    print ('Starting')
    with Progress():
        for i in range(5):
            time.sleep(3)
    print ('Done')

