from tensorflow.keras import optimizers
import numpy as np
class Optimizer():
    __optimizer_name = None
    __learning_rate = None
    __rho = 0.9
    __momentum = None
    __epsilon = 1e-07
    __clipnorm = None
    __clipvalue = None
    __decay = None
    __lr_sch = None
    def __init__(self, optimizer_name='RMSprop', lr_sch=False, learning_rate=0.001, momentum=0.89, clipnorm=0.89, clipvalue=0.5, decay=10e-06):
        super().__init__()
        self.__optimizer_name = optimizer_name
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__clipnorm = clipnorm
        self.__clipvalue = clipvalue
        self.__decay = decay
        self.__lr_sch = lr_sch


    def optimizer_choose(self):
        if self.__optimizer_name == 'RMSprop':
            return optimizers.RMSprop(learning_rate=self.__learning_rate, rho=self.__rho, epsilon=self.__epsilon, momentum=self.__momentum,
            clipnorm=self.__clipnorm, clipvalue=self.__clipvalue, decay=self.__decay)
        if self.__optimizer_name == 'SGD' and self.__lr_sch == False :
            return optimizers.SGD(learning_rate=self.__learning_rate, momentum=self.__momentum, nesterov=True, clipnorm=self.__clipnorm, clipvalue=self.__clipvalue, decay=self.__decay)
        else:
            return optimizers.SGD(momentum=self.__momentum, nesterov=True, clipnorm=self.__clipnorm, clipvalue=self.__clipvalue, decay=self.__decay)
    
    def exp_decay(self, epoch):
        initial_lrate = self.__learning_rate
        k = 0.15
        lrate = initial_lrate * np.exp(-k*epoch)
        return lrate