import numpy as np

import metrics

# TODO : add conv2d layer
# TODO : max-pool layer
# TODO : add dropout layer, remove lambda if dropout is used
# TODO : add batch and layer normalization (for dense)
# TODO : add spatial batch and group normalization (for conv2d)

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

        K = np.sqrt(1 / input_dim)
        if self.activation is metrics.Softmax :  # Xavier initialization for Softmax activated layers
            self.weights = np.random.uniform(-K, K, size=(input_dim, self.num_nodes))
        elif self.activation is metrics.ReLU :  # He initialization for ReLU activated layers
            self.weights = np.random.normal(scale=np.sqrt(2) * K, size=(input_dim, self.num_nodes))

    # forward pass
    def forward( self, datapoints, loss=None ) :
        # input array can only be C-vector or N x C array, where C - no. of features and N - no. of datapoints
        assert datapoints.ndim <= 2, \
            "Expected input to Dense layer is at most a 2D numpy array."

        self.input = datapoints  # caching input data
        self.nodes = self.activation.forward(datapoints @ self.weights + self.bias)
        if loss is None : return self.nodes  # loss calculation is unnecessary for prediction

        loss += 0.5 * (  # regularization term, lambda multiplied at the end
            np.sum(np.square(self.weights)) +
            np.sum(np.square(self.bias))
        )
        return self.nodes, loss

    # backpropagation step
    def backward( self, gradients, reg_lambda, learning_rate ) :
        gradients = self.activation.backward(gradients, self.nodes)  # gradient wrt activation function

        self.bias -= learning_rate * (np.mean(gradients, axis=0) + reg_lambda * self.bias)

        weight_grads = np.mean(  # gradients for updating weights
            self.input[:, :, np.newaxis] @ gradients[:, np.newaxis, :],
            axis=0
        )
        weight_grads += reg_lambda * self.weights  # gradient of regularization term
        gradients = gradients @ self.weights.T  # gradient for passing to previous layer
        self.weights -= learning_rate * weight_grads  # updating weights

        return gradients

    # display the details of the layer
    def details( self ) :
        print(
            f"|{type(self).__name__:^11}|{str(self.weights.shape):^17}|"
            f"{self.weights.size:^14}|{self.activation.__name__:^14}|"
        )


# Represents a Neural Network, which can hold multiple layers and a loss metric
class NeuralNetwork :
    # TODO : update rule classes (SGDMomentum, RMSProp, Adam) new file called optimizers.py

    def __init__( self, train_data, train_labels, LossModel, Layers ) :
        train_check_split = 1000
        self.check_data = train_data[: train_check_split]  # data for accuracy checking set
        self.check_labels = train_labels[: train_check_split]  # labels for accuracy checking set
        self.train_data = train_data[train_check_split :]  # data for training weights
        self.train_labels = train_labels[train_check_split :]  # labels for training weights
        self.loss_model = LossModel

        prev_output_size = train_data.shape[1]
        self.layers = []  # list of all layers
        for layer in Layers :  # initializing weights of all layers, using previous layer output size
            layer.init_weights(prev_output_size)
            self.layers.append(layer)  # populating layers list
            prev_output_size = layer.num_nodes

    # loads previously saved weights into the network's layers
    def load_weights( self, weights_list ) :
        for layer_num in range(len(weights_list)) :
            self.layers[layer_num].weights, self.layers[layer_num].bias = weights_list[layer_num]

    # train the weights of the model using a dataset and other parameters
    def train( self, epochs, batch_size, learning_rate, lr_decay=0.5 ) :
        EPOCHS_FOR_DECAY = 2  # number of epochs to decay learning rate at
        num_data = self.train_data.shape[0]  # number of datapoints in training data
        iterations_per_epoch = num_data // batch_size  # number of iterations per epoch
        num_iterations = int(epochs) * iterations_per_epoch  # total number of iterations

        best_weights = None  # stores the best weights so far every epoch
        best_accuracy = 0  # metric to update best weights
        loss_iterations = np.zeros(num_iterations)  # stores the loss values over the iterations
        for itr in range(num_iterations) :
            # extracting a random batch from the full dataset, with replacement
            batch = np.random.choice(num_data, batch_size)
            labels_batch = self.train_labels[batch]

            # calculate final scores and loss
            scores, loss_iterations[itr] = self.forward(self.train_data[batch], labels_batch)
            """sanity check using numerical gradient calculation
            num_gradients = [
                (metrics.gradient_check(self, data_batch, layer.weights, labels_batch),
                 metrics.gradient_check(self, data_batch, layer.bias, labels_batch))
                for layer in self.layers
            ]  # """
            self.backward(scores, labels_batch, learning_rate)  # update weights of all layers

            # every epoch, check goodness of current weights and update best weights if current weights are better
            if (itr + 1) % iterations_per_epoch == 0 :
                check_accuracy = np.sum(self.check_labels == self.predict(self.check_data)) / 10
                if check_accuracy > best_accuracy :
                    best_accuracy = check_accuracy
                    best_weights = [(layer.weights, layer.bias) for layer in self.layers]

                # decay learning rate every fixed number of epochs
                if (itr + 1) % (EPOCHS_FOR_DECAY * iterations_per_epoch) == 0 :
                    learning_rate *= lr_decay

        self.load_weights(best_weights)  # load the best weights of the entire training session
        return loss_iterations, best_weights

    # predict an output class for given input datapoint(s)
    def predict( self, test_points ) :
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
        print("\nModel structure :")
        print(" -" * 30)
        print(
            "|{0:^11}|{1:^17}|{2:^14}|{3:^14}|".format(
                "Layer", "Shape", "Parameters", "Activation"
            )
        )
        print(" -" * 30)
        for layer in self.layers : layer.details()
        print(" -" * 30)
        print("| Input dimension  :  {0:<38}|".format(self.layers[0].weights.shape[0]))
        print("| Output dimension :  {0:<38}|".format(self.layers[-1].num_nodes))
        print(" -" * 30)
