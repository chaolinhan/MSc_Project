import os
import multiprocessing
from pyabc import sge

print("\nos.cpu_count: %d" % os.cpu_count())
print("\nmultiprocessing.cpu_count: %d" % multiprocessing.cpu_count())
print("\nOMP_THREADS: %s" % os.environ["OMP_NUM_THREADS"])
print("\npyABC seg detected: %d " % sge.nr_cores_available())
