import numpy as np


"""For each subject, we acquired images for each
combination of five horizontal head poses (0°, ±15°, ±30°),
seven horizontal gaze directions (0°, ±5°, ±10°, ±15°),
and three vertical gaze directions (0°, ±10°)"""


def vectorized_result(v, h):
    vm = np.zeros((8, 1))
    vertical_index = v // 10 + 1
    horizontal_index = h // 5 + 5
    vm[vertical_index] = 1.0
    vm[horizontal_index] = 1.0
    return vm


def vectorized_result_2(v, h):
    vm = np.zeros((1, 1))
    if abs(h) <= 5 and v == 0:
        vm[0] = 1
    else:
        vm[0] = 0
    return vm
