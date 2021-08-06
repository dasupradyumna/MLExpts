import numpy as np


class AnnealingSchedule :

    def __new__( cls, lr_decay ) :
        if cls is AnnealingSchedule :
            raise TypeError("AnnealingSchedule class must not be instantiated.\n")
        return object.__new__(cls)

    def __init__( self, lr_decay ) :
        self.lr_decay = lr_decay
        self.counter = 0


class StepDecay(AnnealingSchedule) :

    def __call__( self, learning_rate ) :
        return learning_rate * self.lr_decay


class ExpDecay(AnnealingSchedule) :

    def __call__( self, learning_rate ) :
        return learning_rate * np.exp(- self.lr_decay)


class InverseDecay(AnnealingSchedule) :

    def __call__( self, learning_rate ) :
        factor = (1 + self.lr_decay * self.counter) / (1 + self.lr_decay * (self.counter + 1))
        self.counter += 1
        return learning_rate * factor


class SGD :

    def __init__( self, learning_rate, *, decay_type=StepDecay, lr_decay=1 ) :
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.decay = decay_type

    def __call__( self, gradients ) :
        return - self.learning_rate * gradients

    def decay_learning_rate( self ) :
        self.learning_rate = self.decay(self.learning_rate)


class SGDMomentum(SGD) :

    def __init__( self, learning_rate, *, decay_type=StepDecay, lr_decay=1, friction=0.95 ) :
        super().__init__(learning_rate, decay_type=decay_type, lr_decay=lr_decay)
        self.velocity = 0
        self.friction = friction

    def reset( self ) :
        self.velocity = 0

    def __call__( self, gradients ) :
        self.velocity = self.friction * self.velocity - self.learning_rate * gradients
        return self.velocity


class NAG(SGDMomentum) :

    def __call__( self, gradients ) :
        velocity_old = self.velocity
        self.velocity = self.friction * self.velocity - self.learning_rate * gradients
        return (1 + self.friction) * self.velocity - self.friction * velocity_old
