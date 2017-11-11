import random

def gen_random_pairs(elems):
	i = 0
	l = len(elems)-1
	random.shuffle(elems)
	while True:
		e = elems[i]
		i = i + 1
		if i > l:
			random.shuffle(elems)
			i = 0
		e2 = elems[i]
		i = i + 1
		if i > l:
			random.shuffle(elems)
			i = 0
		yield e, e2