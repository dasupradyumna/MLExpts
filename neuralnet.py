import numpy as np


class NNLayer :  # for now, this exists only to group all layers as its subclasses
    pass


class Dense(NNLayer) :

    def __init__( self, NumNodes, Activation ) :
        self.activation = Activation
        self.weights = None
        self.nodes = np.zeros(NumNodes)

    def init_weights( self, input_dim ) :
        self.weights = 1e-3 * np.random.randn(input_dim, self.nodes.size)  # gaussian initialization

    def forward( self, datapoints ) :
        self.nodes = self.activation(datapoints @ self.weights)
        return self.nodes

    def backward( self ) :
        pass

    def view( self ) :
        pass


class NeuralNetwork :

    def __init__( self, LossModel, InputDim, Layers ) :
        self.loss_model = LossModel

        prev_output_size = InputDim
        self.layers = []
        for layer in Layers :
            assert issubclass(type(layer), NNLayer), \
                f"NeuralNetwork constructor argument is not an NNLayer object.\n{repr(layer)}"
            layer.init_weights(prev_output_size)
            self.layers.append(layer)
            prev_output_size = layer.num_nodes

    def forward( self, datapoints ) :
        nodes = datapoints
        for layer in self.layers :
            nodes = layer.forward(nodes)
        return nodes

    def backward( self ) :
        for layer in self.layers : layer.backward()

    def view( self ) :
        for layer in self.layers : layer.view()
