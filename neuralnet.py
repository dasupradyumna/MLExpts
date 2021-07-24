from math import sqrt

import numpy as np

import metrics


# Represents the most basic, "fully-connected" (or dense) layer for neural networks
class Dense :

    def __init__( self, NumNodes, Activation ) :
        self.num_nodes = NumNodes  # number of nodes in current layer
        self.activation = Activation  # activation function to be applied to output of the affine product
        self.weights = None  # weights matrix connecting to previous layer
        self.bias = None  # bias vector for current layer
        self.nodes = None  # array to store the values of current layer
        self.input = None  # cache of input array to the layer for backpropagation

    # initialize weights of the layer depending upon the activation function
    def init_weights( self, input_dim ) :
        self.bias = np.zeros(self.num_nodes)  # zero-initialize the bias vector

        K = sqrt(1 / input_dim)
        if self.activation is metrics.Softmax :  # Xavier initialization for Softmax activated layers
            self.weights = np.random.uniform(-K, K, size=(input_dim, self.num_nodes))
        elif self.activation is metrics.ReLU :  # He initialization for ReLU activated layers
            self.weights = np.random.normal(scale=sqrt(2) * K, size=(input_dim, self.num_nodes))

    # forward pass
    def forward( self, datapoints, loss=None ) :
        # input array can only be C-vector or N x C array, where C - no. of features and N - no. of datapoints
        assert datapoints.ndim <= 2, \
            "Expected input to Dense layer is at most a 2D numpy array."

        self.input = datapoints  # caching input data
        self.nodes = self.activation.forward(datapoints @ self.weights + self.bias)
        if loss is None : return self.nodes  # loss calculation is unnecessary for prediction

        loss += 0.5 * (  # regularization term, lambda multiplied at the end
            np.sum(self.weights * self.weights) +
            np.sum(self.bias * self.bias)
        )
        return self.nodes, loss

    # backpropagation step
    def backward( self, gradients, reg_lambda, learning_rate ) :
        gradients = self.activation.backward(gradients, self.nodes)  # gradient wrt activation function

        self.bias -= learning_rate * (np.mean(gradients, axis=0) + reg_lambda * self.bias)

        weight_grads = np.mean(  # gradients for updating weights
            self.input.reshape((-1, self.input.shape[1], 1)) @ gradients.reshape((-1, 1, self.num_nodes)),
            axis=0
        )
        weight_grads += reg_lambda * self.weights  # gradient of regularization term
        gradients = gradients @ self.weights.T  # gradient for passing to previous layer
        self.weights -= learning_rate * weight_grads  # updating weights

        return gradients

    # display the details of the layer
    def details( self ) :
        print(
            "|{0:^11}|{1:^17}|{2:^14}|{3:^14}|".format(
                "Dense", str(self.weights.shape), self.weights.size, self.activation.__qualname__
            )
        )


# Represents a Neural Network, which can hold multiple layers and a loss metric
class NeuralNetwork :

    def __init__( self, LossModel, InputDim, Layers ) :
        self.loss_model = LossModel

        prev_output_size = InputDim
        self.layers = []  # list of all layers
        for layer in Layers :  # initializing weights of all layers, using previous layer output size
            layer.init_weights(prev_output_size)
            self.layers.append(layer)  # populating layers list
            prev_output_size = layer.num_nodes

    # train the weights of the model using a dataset and other parameters
    def train( self, datapoints, labels, num_iterations, learning_rate ) :
        # TODO : random batches with replacement
        # TODO : add training epochs, 1 epoch = batch size * iterations per epoch
        # TODO : learning rate decay with epochs
        # TODO : keep track of best weights, end of each epoch
        batch_size = datapoints.shape[0] // num_iterations
        loss_iterations = np.zeros(num_iterations)
        for itr in np.random.permutation(range(num_iterations)) :  # random batch without replacement
            # extracting a batch from the full dataset, using the above iterator
            data_batch = datapoints[itr * batch_size : (itr + 1) * batch_size]
            labels_batch = labels[itr * batch_size : (itr + 1) * batch_size]
            scores, loss_iterations[itr] = self.forward(data_batch, labels_batch)  # calculate final scores and loss
            """sanity check using numerical gradient calculation
            num_gradients = [
                (metrics.gradient_check(self, data_batch, layer.weights, labels_batch),
                 metrics.gradient_check(self, data_batch, layer.bias, labels_batch))
                for layer in self.layers
            ]  # """
            self.backward(scores, labels_batch, learning_rate)  # update weights of all layers

        return loss_iterations

    # predict an output class for given input datapoint(s)
    def predict( self, test_points ) :
        # TODO : add batch size prediction
        nodes = test_points
        for layer in self.layers :  # propagating data forwards through each layer
            nodes = layer.forward(nodes)
        return np.argmax(nodes, axis=-1)  # find index of max score

    # forward pass
    def forward( self, datapoints, labels ) :
        loss = 0  # to accumulate weights regularization term over the layers
        nodes = datapoints
        for layer in self.layers :  # propagating data forwards through each layer
            nodes, loss = layer.forward(nodes, loss)
        loss = self.loss_model.forward(nodes, labels, loss)  # calculate loss using output of the last layer
        return nodes, loss

    # backpropagation step
    def backward( self, scores, labels, learning_rate ) :
        gradients, reg_lambda = self.loss_model.backward(scores, labels)  # gradient wrt output of last layer
        for layer in reversed(self.layers) :  # propagating gradients backwards through each layer
            gradients = layer.backward(gradients, reg_lambda, learning_rate)

    # display the details of the network's structure
    def details( self ) :
        print("Model structure :")
        print(' -' * 30)
        print(
            "|{0:^11}|{1:^17}|{2:^14}|{3:^14}|".format(
                "Layer", "Shape", "Parameters", "Activation"
            )
        )
        print(' -' * 30)
        for layer in self.layers : layer.details()
        print(' -' * 30)
        print("| Input dimension  :  {0:<38}|".format(self.layers[0].weights.shape[0]))
        print("| Output dimension :  {0:<38}|".format(self.layers[-1].num_nodes))
        print(' -' * 30)
