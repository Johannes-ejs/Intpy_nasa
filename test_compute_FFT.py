from __future__ import print_function
import time
import numpy as np
import numpy.random as rn
import sys
from intpy.intpy import initialize_intpy, deterministic

@deterministic
def compute_FFT(n):
    A= np.eye(n) + 1j *np.eye (n)
    result = np.fft.fft2(A)
    return np.abs(result)

@initialize_intpy(__file__)
def main(n):
    print('FFT: ', n)
    compute_FFT(n)



if __name__ == "__main__":
    N = int(sys.argv[1])
    start = time.perf_counter()
    main(N)
    print(time.perf_counter()-start)
