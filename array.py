import numpy as np


def vectorized_result(v, h):
    vm = np.zeros((8, 1))
    vertical_index = v // 10 + 1
    horizontal_index = h // 5 + 5
    vm[vertical_index] = 1.0
    vm[horizontal_index] = 1.0
    return vm
