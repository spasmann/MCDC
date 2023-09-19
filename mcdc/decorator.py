import time
import numpy as np
from numba import njit, objmode

results = {}
results2 = {}
tree = {"stack": ["main"], "main": set()}
USE_TIMER = True


def get_decorator():
    if USE_TIMER:
        return njit_timer
    else:
        return njit


def print_tree(node, layer):
    for n in node:
        print("{:.6f}".format(np.min(results[n])), "-|-" * layer, n)
        print_tree(tree[n], layer + 1)


def function_start(f):
    start = time.time()
    tree[tree["stack"][-1]].add(f.__name__)
    tree["stack"] += [f.__name__]
    if f.__name__ not in results:
        tree[f.__name__] = set()
        results[f.__name__] = list()
        results2[f.__name__] = list()
    return start


def function_end(f, start):
    run_time = time.time() - start
    results[f.__name__] += [run_time]
    results2[f.__name__] += [run_time]
    if len(tree["stack"]) > 2:
        results2[tree["stack"][-2]] += [-run_time]
    tree["stack"] = tree["stack"][:-1]


def njit_timer(f):
    jf = njit(f)

    @njit
    def wrapper(*args):
        with objmode(start="float64"):
            start = function_start(f)
        g = jf(*args)
        with objmode():
            function_end(f, start)
        return g

    return wrapper


def python_timer(f):
    def wrapper(*args):
        start = function_start(f)
        g = f(*args)
        function_end(f, start)
        return g

    return wrapper
