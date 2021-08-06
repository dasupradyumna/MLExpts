from time import time

import numpy as np
# 32x32x3 color images of 10 classes of objects : 50000 training points, 10000 testing points
from tensorflow.keras.datasets import cifar10
# 28x28x1 gray images of handwritten digits (10 classes) : 60000 training points, 10000 testing points
from tensorflow.keras.datasets import mnist

import features
import metrics
import optimizers
from classifier import KNearestNeighbour, Linear
from neuralnet import Dense, NeuralNetwork


def testKNN( ) :
    def preprocess( dataset ) :
        # subsample to 14x14 (not doing this increases execution time by 4 times, since images will be 4 times bigger)
        num_data, img_size = dataset.shape
        dataset = dataset.reshape(-1, img_size // 2, 2, img_size // 2, 2).max(axis=(2, 4)).copy()  # contiguous array
        return dataset.reshape(num_data, -1)  # flatten 14x14 to 196

    ############ UNPACKING AND PREPROCESSING DATASET ############

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


def __preprocess( images, labels, feature_generator=None ) :
    # images : N x H x W x C, no. of images, height, width, channels of each image
    # labels : N x 1, true labels for each image
    if images.ndim == 3 : images = images[..., np.newaxis]
    N, H, W, C = images.shape
    labels.resize(N)  # making labels an N-vector
    images = images / 255  # normalize pixel values to [0,1] interval
    if feature_generator is None :
        images = images.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))  # subsample to 1/4th of original image
        images -= np.mean(images, axis=(1, 2), keepdims=True)  # subtracting mean per channel per image
        return images.reshape(N, -1), labels  # N x K, K features and labels
    else :
        return feature_generator(images, orientations=4, cell_size=4), labels


def testLinear( ) :
    start = time()
    NUM_CLASSES = 10
    NUM_ITERATIONS = 1000

    ############ UNPACKING AND PREPROCESSING DATASET ############

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, train_labels = __preprocess(train_images, train_labels)
    test_images, test_labels = __preprocess(test_images, test_labels)
    half_test = test_labels.size // 2  # splitting half of test data for validation
    validation_images, validation_labels = test_images[: half_test], test_labels[: half_test]
    test_images, test_labels = test_images[half_test :], test_labels[half_test :]
    print("Datasets loaded and preprocessed.\n")

    ############ TUNING HYPERPARAMETERS AND VALIDATION ############

    linear_classifier = Linear(NUM_CLASSES)

    hyperparameters = np.zeros((4, 6, 2))
    # hyperparameters[:, :, :, 0] = np.array([0.01, 0.05, 0.1, 0.5]).reshape((4, 1, 1))  # loss_margin
    hyperparameters[:, :, 0] = np.array([0.5, 0.1, 1e-2, 1e-3])[:, np.newaxis]  # learning_rate
    hyperparameters[:, :, 1] = np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])[np.newaxis, :]  # reg_lambda
    hyperparameters.resize(24, 2)

    print("------- Validation -------")
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

    print("\n------- Testing -------")
    learning_rate, reg_lambda = hyperparameters[np.argmax(accuracy_hp)]  # best configuration
    # linear_classifier.setLoss(metrics.MultiSVMLoss(reg_lambda, loss_margin))  # 31~32% average test accuracy
    linear_classifier.setLoss(metrics.SparseCELoss(reg_lambda))  # 32~33% average test accuracy
    linear_classifier.train(train_images, train_labels, NUM_ITERATIONS, learning_rate)
    prediction = linear_classifier.predict(test_images)
    correct = np.sum(prediction == test_labels)  # counting no. of correct predictions

    print(f"Total Time elapsed : {time() - start}s")
    print(f"Accuracy = {correct}/{len(test_labels)} : {100 * correct / len(test_labels)}%")


def testNN( ) :
    ############ UNPACKING AND PREPROCESSING DATASET ############

    NUM_CLASSES = 10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_features, train_labels = __preprocess(train_images, train_labels, features.HOG)
    test_features, test_labels = __preprocess(test_images, test_labels, features.HOG)
    half_test = test_labels.size // 2  # splitting half of test data for validation
    validation_features, validation_labels = test_features[: half_test], test_labels[: half_test]
    test_features, test_labels = test_features[half_test :], test_labels[half_test :]
    del train_images, test_images
    print("Datasets loaded and preprocessed.\n")

    ############ TUNING HYPERPARAMETERS AND VALIDATION ############

    # [ learning_rate, reg_lambda, EPOCHS ]
    hyperparameters = np.zeros((3, 3, 2, 3))
    hyperparameters[..., 0] = np.array([0.5, 0.25, 0.1])[:, np.newaxis, np.newaxis]
    hyperparameters[..., 1] = np.array([1e-4, 1e-6, 1e-8])[np.newaxis, :, np.newaxis]
    hyperparameters[..., 2] = np.array([8, 10])[np.newaxis, np.newaxis, :]
    hyperparameters.resize(18, 3)
    BATCH_SIZE = 100

    print("------- Training / Validation -------")
    print(
        "  N  | {0:^6} {1:^8} {2:^6} | {3:^8} ,  {4} ".format(
            "LR", "Lambda", "Epochs", "Accuracy", "Time taken"
        )
    )
    print("--------------------------------------------------------")

    Model = NeuralNetwork(
        train_features, train_labels,
        Layers=(
            Dense(64, metrics.ReLU),
            Dense(32, metrics.ReLU),
            Dense(NUM_CLASSES, metrics.Softmax)
        )
    )
    accuracy_hp = []
    best_weights_hp = []
    loss_hp = []
    count = 1
    for learning_rate, reg_lambda, EPOCHS in hyperparameters :
        start = time()
        Model.loss_model = metrics.SparseCELoss(reg_lambda)
        Model.update_rule = optimizers.SGD(learning_rate, train_features.shape[0] // BATCH_SIZE, lr_decay=0.5)

        loss, best_weights = Model.train(EPOCHS, BATCH_SIZE)
        best_weights_hp.append(best_weights)
        loss_hp.append(loss)
        Model.load_weights(best_weights)
        predictions = Model.predict(validation_features)
        correct = np.sum(predictions == validation_labels)
        accuracy_hp.append(100 * correct / len(validation_labels))  # log accuracy for current configuration

        print(
            f"{count:^5}| {learning_rate:^6} {reg_lambda:^8} {int(EPOCHS):^6} "
            f"| {accuracy_hp[-1]:>6} % ,  {(time() - start):.3f}s"
        )
        count += 1

    ############ PREDICTION WITH WEIGHTS CORRESPONDING TO BEST HYPERPARAMETERS  ############

    Model.details()
    print("\n------- Testing -------")
    best_accuracy = np.argmax(accuracy_hp)
    Model.load_weights(best_weights_hp[best_accuracy])
    predictions = Model.predict(test_features)
    correct = np.sum(predictions == test_labels)
    print(f"Accuracy = {correct}/{len(test_labels)} = {100 * correct / len(test_labels)}%")


if __name__ == '__main__' :
    # testKNN()
    # testLinear()
    testNN()
