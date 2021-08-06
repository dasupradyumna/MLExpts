import numpy as np


def step_decay( decay_rate ) :
    if decay_rate < 0 or decay_rate > 1 :
        raise ValueError("decay constant for step decay must be in open (0,1).\n")
    return decay_rate


def exp_decay( decay_rate ) :
    if decay_rate < 0 :
        raise ValueError("decay constant for exponential decay must be non-negative.\n")
    return np.exp(-decay_rate)


class SGD :

    def __init__( self, learning_rate, *, decay_type=step_decay, learning_rate_decay=1 ) :
        self.learning_rate = learning_rate
        self.lr_decay = learning_rate_decay
        self.decay_function = decay_type

    def __call__( self, gradients ) :
        return - self.learning_rate * gradients

    def decay( self ) :
        self.learning_rate *= self.decay_function(self.lr_decay)


class SGDMomentum(SGD) :

    def __init__( self, learning_rate, *, decay_type=step_decay, learning_rate_decay=1, friction=0.95 ) :
        super().__init__(learning_rate, decay_type=decay_type, learning_rate_decay=learning_rate_decay)
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
