import numpy as np
import sys
from intpy.intpy import initialize_intpy, deterministic
import time


#@deterministic
def f(A,x):
    x0 = np.log(np.dot(A, np.exp(x)))
    x0 -= np.log(np.sum(np.exp(x0)))
    return x0

def belief_propagation(N):
    dim = 5000
    A = np.random.rand(dim, dim)
    x = np.ones((dim,))
    for i in range(N):
        x=f(A,x)
    return x


#@initialize_intpy(__file__)
def main(n):    
    belief_propagation(n)


if __name__ == "__main__":
    N = int(sys.argv[1])
    start = time.perf_counter()
    main(N)
    print(time.perf_counter()-start)