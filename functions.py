import numpy as np
import numpy.random as rnd
from numba import njit, jit, prange 




@njit
def f_1(x):
    return np.exp(x)



@njit
def f_2(x):
    return np.square(np.exp(x))



@njit(parallel=True)
def crude_mc_integration(f, monte_carlo_steps, upper_limit, lower_limit): # Checked OK
    sum_f = 0
    sum_f_2 = 0
    multiplier = (upper_limit - lower_limit) / monte_carlo_steps

    for i in prange(monte_carlo_steps):
        x = lower_limit + (rnd.random() * (upper_limit - lower_limit))
        f_x = f(x)

        sum_f += f_x
        sum_f_2 += f_x**2

    result = multiplier * sum_f
    variance = multiplier * sum_f_2 - (result ** 2)

    return result, np.sqrt(variance / monte_carlo_steps)
    