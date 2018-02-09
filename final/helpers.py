import numpy as np
from numba import jit

@jit
def swap(arr, a, b):
	temp = arr[a]
	arr[a] = arr[b]
	arr[b] = temp
	
@jit
def pick_weighted_index(weights):
	# this function was fastest when tested against np and raw python
	size = len(weights)
	total = weights.sum()
	p = np.random.random() * total
	for i in range(size):
		p -= weights[i]
		if p <= 0.0: return i
	return size - 1