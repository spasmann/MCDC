# =============================================================================
# Timing Decorator
# =============================================================================

import time
import numpy as np
from numba import njit, objmode

results = {}
tree = {'stack':['main'], 'main':set()}
USE_TIMER = True 

def get_decorator():
    if USE_TIMER:
        return njit_timer
    else:
        return njit
    
def print_tree(node, layer):
    for n in node:
        print('{:.6f}'.format( np.min(results[n]) ), '-|-'*layer, n)
        print_tree(tree[n], layer+1)
    
def wrapper_objm_start(f):
    start = time.time()
    tree[ tree['stack'][-1] ].add( f.__name__ )
    tree['stack'] += [ f.__name__ ]
    if f.__name__ not in results:
        tree[f.__name__] = set()
        # print(tree['stack'])
    return start

def wrapper_objm_end(f, start):
    run_time = time.time() - start
    if f.__name__ in results:
        results[f.__name__] += [run_time]
    else:
        results[f.__name__] = [run_time]
    tree['stack'] = tree['stack'][:-1]

def njit_timer(f):
    jf = njit(f)
    @njit
    def wrapper(*args):
        with objmode(start='float64'):
            start = wrapper_objm_start(f)
        g = jf(*args)
        with objmode():
            wrapper_objm_end(f, start)
        return g
    return wrapper