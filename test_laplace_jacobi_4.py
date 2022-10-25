#!/usr/bin/env python

################################################
# Numerical solution of the Laplace's equation.
#
#  u   + u   = 0
#   xx    yy
#
# Fourth-order compact Scheme
# Jacobi's iteration
################################################

from __future__ import print_function
import numpy as np
import sys
from intpy.intpy import initialize_intpy, deterministic
import time

def vector_time_step(u):
    u_old = u.copy()
    u[1:-1, 1:-1] = (
                     (u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] +
                      u[1:-1, 2:])*4.0 +
                     u[0:-2, 0:-2] + u[0:-2, 2:] + u[2:, 0:-2] + u[2:, 2:]
    )/20.0

    return u, np.linalg.norm(u-u_old)


@deterministic
def vectorized_solver(n):
    j = complex(0, 1)
    pi_c = np.pi
    u = np.zeros((n, n), dtype=float)
    x = np.r_[0.0:pi_c:n*j]
    u[0, :] = np.sin(x)
    u[n-1, :] = np.sin(x)*np.exp(-pi_c)
    iteration = 0
    error = 2
    while(iteration < 100000 and error > 1e-6):
        (u, error) = vector_time_step(u)
        iteration += 1
    return (u, error, iteration)


@initialize_intpy(__file__)
def main(num_points):
    print("Jacobi solver for Laplace Equation: ", num_points)
    (u, error, iteration) = vectorized_solver(num_points)


if __name__ == "__main__":
    N = int(sys.argv[1])
    start = time.perf_counter()
    main(N)
    print(time.perf_counter()-start)


