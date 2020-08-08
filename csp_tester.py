from csp import *
import time
import itertools
import numpy as np


def test_solver_on_pics(pics, solver):
    start = time.time()
    sol = csp_solver_factory(pics, solver)
    return time.time() - start, sol.get_answer()


def test_solver(pics, solver):
    total_time = 0
    success_count = 0
    tests_number = 0
    for comb in itertools.combinations_with_replacement(pics, 3):
        tests_number += 1
        t, sol = test_solver_on_pics(comb, solver)
        total_time += t
        if sol is not None:
            success_count += 1
    avg_time = total_time / tests_number
    success_rate = success_count / tests_number
    return avg_time, success_rate


def test_pics(pics):
    pass