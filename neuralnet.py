from math import sqrt

import numpy as np

import metrics


class Dense :

    def __init__( self, NumNodes, Activation ) :
        self.num_nodes = NumNodes
        self.activation = Activation
        self.weights = None
        self.nodes = None
        self.input = None

    def init_weights( self, input_dim ) :
        K = sqrt(1 / input_dim)
        if self.activation is metrics.Softmax :  # Xavier initialization
            self.weights = np.random.uniform(-K, K, size=(input_dim, self.num_nodes))
        elif self.activation is metrics.ReLU :  # He initialization
            self.weights = np.random.normal(scale=sqrt(2) * K, size=(input_dim, self.num_nodes))

    def forward( self, datapoints, loss=None ) :
        assert datapoints.ndim <= 2, \
            "Expected input to Dense layer is at most a 2D numpy array."

        self.input = datapoints
        self.nodes = self.activation.forward(datapoints @ self.weights)
        if loss is None : return self.nodes

        loss += 0.5 * np.sum(self.weights * self.weights)
        return self.nodes, loss

    def backward( self, gradients, reg_lambda, learning_rate ) :
        gradients = self.activation.backward(gradients, self.nodes)
        weight_grads = np.mean(
            self.input.reshape((-1, self.input.shape[1], 1)) @ gradients.reshape((-1, 1, self.num_nodes)),
            axis=0
        )
        weight_grads += reg_lambda * self.weights
        gradients = gradients @ self.weights.T
        self.weights -= learning_rate * weight_grads
        return gradients

    def view( self ) :
        pass


class NeuralNetwork :

    def __init__( self, LossModel, InputDim, Layers ) :
        self.loss_model = LossModel

        prev_output_size = InputDim
        self.layers = []
        for layer in Layers :
            layer.init_weights(prev_output_size)
            self.layers.append(layer)
            prev_output_size = layer.num_nodes

    def train( self, datapoints, labels, num_iterations, learning_rate ) :
        batch_size = datapoints.shape[0] // num_iterations
        loss_iterations = np.zeros(num_iterations)
        for itr in np.random.permutation(range(num_iterations)) :  # random batch without replacement
            data_batch = datapoints[itr * batch_size : (itr + 1) * batch_size]
            labels_batch = labels[itr * batch_size : (itr + 1) * batch_size]
            scores, loss_iterations[itr] = self.forward(data_batch, labels_batch)
            self.backward(scores, labels_batch, learning_rate)
            """
            # sanity check using numerical gradient calculation
            import metrics
            num_gradients = metrics.gradient_check(self.weights, datapoints, labels, self.loss_model)
            """

        return loss_iterations

    def predict( self, test_point ) :
        assert test_point.ndim <= 2, \
            "Expected input to Dense layer is at most a 2D numpy array."

        nodes = test_point
        for layer in self.layers :
            nodes = layer.forward(nodes)
        axis = int(nodes.ndim == 2)
        return np.argmax(nodes, axis)

    def forward( self, datapoints, labels ) :
        loss = 0
        nodes = datapoints
        for layer in self.layers :
            nodes, loss = layer.forward(nodes, loss)
        loss = self.loss_model.forward(nodes, labels, loss)
        return nodes, loss

    def backward( self, scores, labels, learning_rate ) :
        gradients, reg_lambda = self.loss_model.backward(scores, labels)
        for layer in reversed(self.layers) :
            gradients = layer.backward(gradients, reg_lambda, learning_rate)

    def view( self ) :
        for layer in self.layers : layer.view()
