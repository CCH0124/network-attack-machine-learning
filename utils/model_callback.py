import numpy as np
import tensorflow as tf

def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
	return lrate


def exp_decay(epoch, lr):
    initial_lrate = lr
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

class MyStopTrainCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('loss') < 0.04):
            print("\nLoss is low so cancelling training!!")
            self.model.stop_training = True