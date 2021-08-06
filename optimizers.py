import numpy as np


class AnnealingSchedule :

    def __new__( cls, lr_decay ) :
        if cls is AnnealingSchedule :
            raise TypeError("AnnealingSchedule class must not be instantiated.\n")
        return object.__new__(cls)

    def __init__( self, lr_decay ) :
        self.lr_decay = lr_decay
        self.epochs = 0


class StepDecay(AnnealingSchedule) :
    EPOCHS_FOR_DECAY = 2

    def __call__( self, learning_rate ) :
        self.epochs += 1
        if self.epochs % self.EPOCHS_FOR_DECAY == 0 :
            return learning_rate * self.lr_decay
        else :
            return learning_rate


class ExpDecay(AnnealingSchedule) :

    def __call__( self, learning_rate ) :
        return learning_rate * np.exp(-self.lr_decay)


class InverseDecay(AnnealingSchedule) :

    def __call__( self, learning_rate ) :
        factor = (1 + self.lr_decay * self.epochs) / (1 + self.lr_decay * (self.epochs + 1))
        self.epochs += 1
        return learning_rate * factor


class SGD :

    def __init__( self, learning_rate, iterations_per_epoch, *, decay_type=StepDecay, lr_decay=1 ) :
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.decay = decay_type
        self.iterations = 0
        self.iter_epoch = iterations_per_epoch

    def __call__( self, gradients ) :
        self.iterations += 1
        if self.iterations % self.iter_epoch == 0 : self._decay_learning_rate()
        return -self.learning_rate * gradients

    def _decay_learning_rate( self ) :
        self.learning_rate = self.decay(self.learning_rate)


class SGDMomentum(SGD) :

    def __init__( self, learning_rate, iterations_per_epoch, *, decay_type=StepDecay, lr_decay=1, friction=0.95 ) :
        super().__init__(learning_rate, iterations_per_epoch, decay_type=decay_type, lr_decay=lr_decay)
        self.velocity = 0
        self.friction = friction

    def __call__( self, gradients ) :
        self.iterations += 1
        if self.iterations % self.iter_epoch == 0 : self._decay_learning_rate()
        self.velocity = self.friction * self.velocity - self.learning_rate * gradients
        return self.velocity


class NAG(SGDMomentum) :

    def __call__( self, gradients ) :
        self.iterations += 1
        if self.iterations % self.iter_epoch == 0 : self._decay_learning_rate()
        velocity_old = self.velocity
        self.velocity = self.friction * self.velocity - self.learning_rate * gradients
        return (1 + self.friction) * self.velocity - self.friction * velocity_old


class AdaGrad :

    def __init__( self, learning_rate=1e-2 ) :
        self.learning_rate = learning_rate
        self.cache = 0

    def __call__( self, gradients ) :
        self.cache += gradients * gradients
        return -self.learning_rate * gradients / (np.sqrt(self.cache) + 1e-8)


class RMSProp :

    def __init__( self, learning_rate=1e-3, lr_decay=0.9 ) :
        if not 0 < lr_decay < 1 :
            raise ValueError("Decay rates must be in open (0,1).\n")
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.cache = 0

    def __call__( self, gradients ) :
        self.cache = self.lr_decay * self.cache + (1 - self.lr_decay) * gradients * gradients
        return -self.learning_rate * gradients / (np.sqrt(self.cache) + 1e-8)


class Adam :

    def __init__( self, learning_rate, decay1=0.9, decay2=0.999 ) :
        if not 0 < decay1 < 1 or not 0 < decay2 < 1 :
            raise ValueError("Decay rates must be in open (0,1).\n")
        self.learning_rate = learning_rate
        self.decay1 = decay1
        self.decay2 = decay2
        self.moment1 = 0
        self.moment2 = 0
        self.iterations = 0

    def __call__( self, gradients ) :
        self.iterations += 1
        self.moment1 = self.decay1 * self.moment1 + (1 - self.decay1) * gradients
        m1_corrected = self.moment1 / (1 - np.power(self.decay1, self.iterations))
        self.moment2 = self.decay2 * self.moment2 + (1 - self.decay2) * gradients * gradients
        m2_corrected = self.moment2 / (1 - np.power(self.decay2, self.iterations))
        return -self.learning_rate * m1_corrected / (np.sqrt(m2_corrected) + 1e-8)
