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

    def preprocess( train, test ) :  # flatten and zero-center both the datasets, padded with column of ones for bias
        train = train.reshape(train.shape[0], -1).astype(float)
        test = test.reshape(test.shape[0], -1).astype(float)
        mean_image = np.mean(np.vstack((train, test)), axis=0)
        train -= mean_image
        test -= mean_image
        train = np.hstack((train, np.ones((train.shape[0], 1))))
        test = np.hstack((test, np.ones((test.shape[0], 1))))
        return train, test

    from time import time
    start = time()
    NUM_CLASSES = 10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = preprocess(train_images, test_images)
    train_labels = train_labels.reshape(train_labels.size)
    test_labels = test_labels.reshape(test_labels.size)

    # linear_classifier = Linear(NUM_CLASSES, metrics.SoftmaxLoss)
    linear_classifier = Linear(NUM_CLASSES, metrics.MultiSVMLoss)
    linear_classifier.train(train_images, train_labels, learning_rate=1e-7, num_iterations=1000)
    # TODO : split half of test dataset into validation dataset, add code for tuning hyper-parameters

    correct = 0
    for image, label in zip(test_images, test_labels) :
        prediction = linear_classifier.predict(image)
        correct += (label == prediction)

    print(f"Time elapsed : {time() - start}s")
    print(f"Accuracy = {correct}/{len(test_labels)} : {100 * correct / len(test_labels)}%")


if __name__ == '__main__' :
    # testKNN()
    testLinear()
