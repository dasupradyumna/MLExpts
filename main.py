import numpy as np

import metrics
from classifier import KNearestNeighbour, Linear


def testKNN( ) :
    # 28x28x1 gray images of handwritten digits (10 classes) : 60000 training points, 10000 testing points
    from tensorflow.keras.datasets import mnist

    def preprocess( dataset ) :
        # subsample to 14x14 (not doing this increases execution time by 4 times, since images will be 4 times bigger)
        img_size = dataset.shape[1]
        dataset = dataset.reshape(-1, img_size // 2, 2, img_size // 2, 2).max(axis=(2, 4))
        return dataset.reshape(dataset.shape[0], -1)  # flatten 14x14 to 196

    from time import time
    start = time()
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = preprocess(train_images)
    test_images = preprocess(test_images)

    knn_classifier = KNearestNeighbour(
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
    half_test = test_labels.size // 2
    validation_images, validation_labels = test_images[: half_test], test_labels[: half_test]
    test_images, test_labels = test_images[half_test :], test_labels[half_test :]

    ############ CREATING CLASSIFIER OBJECT ############

    # linear_classifier = Linear(NUM_CLASSES, metrics.SoftmaxLoss)
    linear_classifier = Linear(NUM_CLASSES, metrics.MultiSVMLossWithGrad)  # ~34% accuracy, with best hyperparameters

    ############ TUNING HYPERPARAMETERS AND VALIDATION ############

    hyperparameters = np.zeros((4, 3, 2, 3))
    hyperparameters[:, :, :, 0] = np.array([0.01, 0.05, 0.1, 0.5]).reshape((4, 1, 1))  # loss_margin (0.05)
    hyperparameters[:, :, :, 1] = np.array([1e-3, 1e-4, 1e-5]).reshape((1, 3, 1))  # learning_rate (1e-4)
    hyperparameters[:, :, :, 2] = np.array([1e-4, 1e-5]).reshape((1, 1, 2))  # reg_lambda (1e-5)
    hyperparameters = hyperparameters.reshape((-1, 3))

    accuracy_hp = []
    for loss_margin, learning_rate, reg_lambda in hyperparameters :
        linear_classifier.train(
            train_images, train_labels, loss_margin,  # train the weights with current configuration
            NUM_ITERATIONS, learning_rate, reg_lambda
        )
        prediction = linear_classifier.predict(validation_images)
        correct = np.sum(prediction == validation_labels)
        accuracy_hp.append(100 * correct / len(validation_labels))  # log accuracy for current configuration

        linear_classifier.weights = None  # reset for fresh training

    ############ TRAINING WITH BEST HYPERPARAMETERS AND PREDICTION ############

    loss_margin, learning_rate, reg_lambda = hyperparameters[np.argmax(accuracy_hp)]  # choose best configuration
    linear_classifier.train(
        train_images, train_labels, loss_margin,
        NUM_ITERATIONS, learning_rate, reg_lambda
    )
    prediction = linear_classifier.predict(validation_images)
    correct = np.sum(prediction == validation_labels)  # counting no. of correct predictions

    print(f"Time elapsed : {time() - start}s")
    print(f"Accuracy = {correct}/{len(test_labels)} : {100 * correct / len(test_labels)}%")


if __name__ == '__main__' :
    # testKNN()
    testLinear()
