import numpy as np
def exp_decay(self, epoch):
    initial_lrate = self.__learning_rate
    k = 0.15
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate