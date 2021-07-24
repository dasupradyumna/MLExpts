import numpy as np

import metrics
from classifier import KNearestNeighbour, Linear
from neuralnet import Dense, NeuralNetwork


def testKNN( ) :
    # 28x28x1 gray images of handwritten digits (10 classes) : 60000 training points, 10000 testing points
    from tensorflow.keras.datasets import mnist

    def preprocess( dataset ) :
        # subsample to 14x14 (not doing this increases execution time by 4 times, since images will be 4 times bigger)
        img_size = dataset.shape[1]
        dataset = dataset.reshape(-1, img_size // 2, 2, img_size // 2, 2).max(axis=(2, 4))
        return dataset.reshape(dataset.shape[0], -1)  # flatten 14x14 to 196

    ############ UNPACKING AND PREPROCESSING DATASET ############

    from time import time
    start = time()
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = preprocess(train_images)
    test_images = preprocess(test_images)

    ############ CREATING CLASSIFIER AND PREDICTION ############

    knn_classifier = KNearestNeighbour(  # there is no concept of training for KNN
        10, (train_images, train_labels), metrics.ManhattanNorm
    )  # L1 norm : 81.34% accuracy, ~276s
    # knn_classifier = KNearestNeighbour(
    #     10, (train_images, train_labels), metrics.EuclideanNorm
    # )  # L2 norm : 32.55% accuracy, ~282s
    correct = 0
    for image, label in zip(test_images, test_labels) :
        prediction = knn_classifier(image)
        correct += (label == prediction)

    print(f"Time elapsed : {time() - start}s")
    print(f"Accuracy = {correct}/{len(test_labels)} : {100 * correct / len(test_labels)}%")


def testLinear( ) :
    # 32x32x3 color images of 10 classes of objects : 50000 training points, 10000 testing points
    from tensorflow.keras.datasets import cifar10

    def preprocess( images, labels ) :
        # images : N x H x W x C, no. of images, height, width, channels of each image
        # labels : N x 1, true labels for each image
        N = images.shape[0]
        images = images / 255  # normalize pixel values to [0,1] interval
        images -= np.mean(images, axis=(1, 2)).reshape((N, 1, 1, -1))  # N x 1 x 1 x C : mean per channel per image
        images = images.reshape((N, -1))  # N x K, K features (all pixels)
        images = np.hstack((images, np.ones((N, 1))))  # appending a column of 1s for including bias in weights
        labels = labels.reshape(N)  # making labels an N-vector
        return images, labels

    from time import time
    start = time()
    NUM_CLASSES = 10
    NUM_ITERATIONS = 1000

    ############ UNPACKING AND PREPROCESSING DATASET ############

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, train_labels = preprocess(train_images, train_labels)
    test_images, test_labels = preprocess(test_images, test_labels)
    half_test = test_labels.size // 2  # splitting half of test data for validation
    validation_images, validation_labels = test_images[: half_test], test_labels[: half_test]
    test_images, test_labels = test_images[half_test :], test_labels[half_test :]
    print('Datasets loaded and preprocessed.\n')

    ############ TUNING HYPERPARAMETERS AND VALIDATION ############

    linear_classifier = Linear(NUM_CLASSES)

    hyperparameters = np.zeros((4, 6, 2))
    # hyperparameters[:, :, :, 0] = np.array([0.01, 0.05, 0.1, 0.5]).reshape((4, 1, 1))  # loss_margin
    hyperparameters[:, :, 0] = np.array([0.5, 0.1, 1e-2, 1e-3]).reshape((4, 1))  # learning_rate
    hyperparameters[:, :, 1] = np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]).reshape((1, 6))  # reg_lambda
    hyperparameters = hyperparameters.reshape((-1, 2))

    print('Validation --')
    accuracy_hp = []
    count = 0
    for learning_rate, reg_lambda in hyperparameters :
        # linear_classifier.setLoss(metrics.MultiSVMLoss(reg_lambda, loss_margin))
        linear_classifier.setLoss(metrics.SparseCELoss(reg_lambda))
        # train the weights with current configuration and predict labels for validation set
        linear_classifier.train(train_images, train_labels, NUM_ITERATIONS, learning_rate)
        prediction = linear_classifier.predict(validation_images)
        correct = np.sum(prediction == validation_labels)
        accuracy_hp.append(100 * correct / len(validation_labels))  # log accuracy for current configuration

        print(count, ' : [ ', learning_rate, ' ', reg_lambda, ' ] : ', accuracy_hp[-1])
        linear_classifier.weights = None  # reset for fresh training
        count += 1

    ############ TRAINING WITH BEST HYPERPARAMETERS AND PREDICTION ############

    print('\nTesting --')
    learning_rate, reg_lambda = hyperparameters[np.argmax(accuracy_hp)]  # best configuration
    # linear_classifier.setLoss(metrics.MultiSVMLoss(reg_lambda, loss_margin))  # 31~32% average test accuracy
    linear_classifier.setLoss(metrics.SparseCELoss(reg_lambda))  # 32~33% average test accuracy
    linear_classifier.train(train_images, train_labels, NUM_ITERATIONS, learning_rate)
    prediction = linear_classifier.predict(test_images)
    correct = np.sum(prediction == test_labels)  # counting no. of correct predictions

    print(f"Total Time elapsed : {time() - start}s")
    print(f"Accuracy = {correct}/{len(test_labels)} : {100 * correct / len(test_labels)}%")


def testNN( ) :
    # 32x32x3 color images of 10 classes of objects : 50000 training points, 10000 testing points
    from tensorflow.keras.datasets import cifar10

    def preprocess( images, labels ) :  # sub-samples to 16x16 images before flattening it
        # images : N x H x W x C, no. of images, height, width, channels of each image
        # labels : N x 1, true labels for each image
        N = images.shape[0]
        W = images.shape[1]
        images = images.reshape(-1, W // 2, 2, W // 2, 2, 3).max(axis=(2, 4))
        images = images / 255  # normalize pixel values to [0,1] interval
        images -= np.mean(images, axis=(1, 2)).reshape((N, 1, 1, -1))  # N x 1 x 1 x C : mean per channel per image
        images = images.reshape((N, -1))  # N x K, K features (all pixels)
        labels = labels.astype(int).reshape(N)  # making labels an N-vector
        return images, labels

    NUM_CLASSES = 10
    NUM_ITERATIONS = 1000

    ############ UNPACKING AND PREPROCESSING DATASET ############

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, train_labels = preprocess(train_images, train_labels)
    test_images, test_labels = preprocess(test_images, test_labels)
    half_test = test_labels.size // 2  # splitting half of test data for validation
    validation_images, validation_labels = test_images[: half_test], test_labels[: half_test]
    test_images, test_labels = test_images[half_test :], test_labels[half_test :]
    print('Datasets loaded and preprocessed.\n')

    ############ TUNING HYPERPARAMETERS AND VALIDATION ############

    hyperparameters = np.zeros((4, 6, 2))
    hyperparameters[:, :, 0] = np.array([1, 0.5, 0.1, 1e-2]).reshape((4, 1))  # learning_rate
    hyperparameters[:, :, 1] = np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]).reshape((1, 6))  # reg_lambda
    hyperparameters = hyperparameters.reshape((-1, 2))

    print('------- Validation -------')
    accuracy_hp = []
    count = 0
    from time import time
    for learning_rate, reg_lambda in hyperparameters :
        start = time()
        NNModel = NeuralNetwork(
            metrics.SparseCELoss(reg_lambda),
            InputDim=train_images.shape[1],
            Layers=[
                Dense(64, metrics.ReLU),
                Dense(32, metrics.ReLU),
                Dense(NUM_CLASSES, metrics.Softmax)
            ]
        )
        NNModel.train(train_images, train_labels, NUM_ITERATIONS, learning_rate)
        predictions = NNModel.predict(validation_images)
        correct = np.sum(predictions == validation_labels)
        accuracy_hp.append(100 * correct / len(validation_labels))  # log accuracy for current configuration

        print(f"{count} : [ {learning_rate} {reg_lambda} ] : {accuracy_hp[-1]}% , {(time() - start):.3f}s")
        count += 1

    ############ TRAINING WITH BEST HYPERPARAMETERS AND PREDICTION ############

    print('\n------- Testing -------')
    start = time()
    learning_rate, reg_lambda = hyperparameters[np.argmax(accuracy_hp)]  # best configuration
    NNModel = NeuralNetwork(
        metrics.SparseCELoss(reg_lambda),
        InputDim=train_images.shape[1],
        Layers=[
            Dense(64, metrics.ReLU),
            Dense(32, metrics.ReLU),
            Dense(NUM_CLASSES, metrics.Softmax)
        ]
    )
    NNModel.details()
    NNModel.train(train_images, train_labels, NUM_ITERATIONS, learning_rate)
    predictions = NNModel.predict(test_images)
    correct = np.sum(predictions == test_labels)

    print(f"Total Time elapsed : {time() - start}s")
    print(f"Accuracy = {correct}/{len(test_labels)} : {100 * correct / len(test_labels)}%")


if __name__ == '__main__' :
    # testKNN()
    # testLinear()
    testNN()
