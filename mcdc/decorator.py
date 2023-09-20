import time
import numpy as np
from mcdc.main import USE_TIMER
from mpi4py import MPI
from numba import njit, objmode

# total time
results = {}
# self time
results2 = {}
tree = {"stack": ["main"], "main": set()}
overhead = 1.07e-7

def get_decorator():
    if USE_TIMER:
        return njit_timer
    else:
        return njit


def print_tree(node, layer):
    for n in node:
        print("{:.6f}".format(np.min(results[n])), "-|-" * layer, n)
        print_tree(tree[n], layer + 1)


def function_start(f_name):
    tree[tree["stack"][-1]].add(f_name)
    tree["stack"] += [f_name]
    if f_name not in results:
        tree[f_name] = set()
        results[f_name] = list()
        results2[f_name] = list()
    start = MPI.Wtime()
    return start


def function_end(f_name, start):
    run_time = MPI.Wtime() - start
    results[f_name] += [run_time]
    results2[f_name] += [run_time]
    if len(tree["stack"]) > 2:
        results2[tree["stack"][-2]] += [-run_time]
    tree["stack"] = tree["stack"][:-1]

compilation = 0.0

def njit_timer(f):
    jf = njit(f)
    
    @njit
    def wrapper(*args):
        with objmode(start="float64"):
            start = function_start(f.__name__)
        g = jf(*args)
        with objmode():
            function_end(f.__name__, start)
        return g
    return wrapper


def python_timer(f):
    def wrapper(*args):
        start = function_start(f.__name__)
        g = f(*args)
        function_end(f.__name__, start)
        return g
    return wrapper


if __name__ == "__main__":    
    N = 10000
    @njit
    def calculate_overhead(N):
        for i in range(N):
            with objmode():
                start = function_start('f_name')
            with objmode():
                function_end('f_name', start)

    calculate_overhead(N)    
    temp = np.array(results['f_name'])
    print('Overhead Average ', temp.mean())
    print('Overhead Standard Dev. ', temp.std())
    
        
