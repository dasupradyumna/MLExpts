import numpy as np


class NNLayer :  # for now, this exists only to group all layers as its subclasses
    pass


class Dense(NNLayer) :

    def __init__( self, NumNodes, Activation ) :
        self.num_nodes = NumNodes
        self.activation = Activation
        self.weights = None
        self.nodes = None

    def init_weights( self, input_dim ) :
        self.weights = 1e-3 * np.random.randn(input_dim, self.num_nodes)  # gaussian initialization

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
        self.weights = []
        for layer in Layers :
            # remove this assert if unnecessary
            assert issubclass(type(layer), NNLayer), \
                f"NeuralNetwork constructor argument is not an NNLayer object.\n{repr(layer)}"

            layer.init_weights(prev_output_size)
            self.layers.append(layer)
            self.weights.append(layer.weights)
            prev_output_size = layer.num_nodes
        self.weights = np.array(self.weights)

    def train( self, datapoints, labels, num_iterations, learning_rate ) :
        num_classes = self.layers[-1].num_nodes
        labels = np.eye(num_classes, dtype=bool)[labels]  # convert label values to one-hot vectors

        batch_size = datapoints.shape[0] // num_iterations
        loss_iterations = np.zeros(num_iterations)
        for itr in np.random.permutation(range(len(num_iterations))) :  # random batch without replacement
            data_batch = datapoints[itr * batch_size : (itr + 1) * batch_size]
            labels_batch = labels[itr * batch_size : (itr + 1) * batch_size]
            predictions = self.forward(data_batch)
            loss_iterations[itr] = self.loss_model.evaluate(self.weights, predictions, labels_batch)
            gradients = self.backward()
            """
            # sanity check using numerical gradient calculation
            import metrics
            num_gradients = metrics.gradient_check(self.weights, datapoints, labels, self.loss_model)
            """
            self.weights -= learning_rate * gradients  # update weights

        return loss_iterations

    def predict( self, test_point ) :
        return self.forward(test_point)

    def forward( self, datapoints ) :
        nodes = datapoints
        for layer in self.layers :
            nodes = layer.forward(nodes)
        return nodes

    def backward( self ) :
        for layer in self.layers : layer.backward()
        return 0

    def view( self ) :
        for layer in self.layers : layer.view()


def relu( x ) :
    return np.maximum(0, x)


def softmax( x ) :
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
