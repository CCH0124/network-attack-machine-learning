
import numpy as np
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
	return lrate


def exp_decay(epoch, lr):
    initial_lrate = lr
    k = 0.15
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate