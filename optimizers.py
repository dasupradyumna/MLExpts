class SGD :

    def __init__( self, learning_rate, learning_rate_decay=1 ) :
        self.learning_rate = learning_rate
        self.lr_decay = learning_rate_decay

    def __call__( self, gradients ) :
        return - self.learning_rate * gradients

    def decay( self ) :
        self.learning_rate *= self.lr_decay


class SGDMomentum(SGD) :

    def __init__( self, learning_rate, learning_rate_decay=1, friction=0.95 ) :
        super().__init__(learning_rate, learning_rate_decay)
        self.velocity = 0
        self.friction = friction

    def reset( self ) :
        self.velocity = 0

    def __call__( self, gradients ) :
        self.velocity = self.friction * self.velocity - self.learning_rate * gradients
        return self.velocity
