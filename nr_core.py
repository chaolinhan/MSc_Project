import os
from pyabc import sge
print("\nos.cpu:")
print(os.cpu_count())
print("\nOMP_THREADS:")
print(os.environ["OMP_NUM_THREADS"])
print("\npyABC seg detected:")
print(sge.nr_cores_available())

