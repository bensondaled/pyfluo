import time
import sys

def display_time_elapsed():
    t0 = time.time()
    while True:
        elap = time.time()-t0
        sys.stdout.write("\r%i seconds elapsed..."%elap)
        sys.stdout.flush()
        time.sleep(0.00001)
