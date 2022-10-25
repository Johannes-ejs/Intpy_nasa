import numpy as np
import sys
from intpy.intpy import initialize_intpy, deterministic
import time

@deterministic
def matrix_multiplication(A, B):
    return np.dot(A,B)


@initialize_intpy(__file__)
def main(n):
    A = np.random.rand(n, n) 
    B = np.random.rand(n, n)
    matrix_multiplication(A, B)

if __name__ == "__main__":
    N = int(sys.argv[1])
    start = time.perf_counter()
    main(N)
    print(time.perf_counter()-start)
