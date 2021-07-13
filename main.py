from classifier import KNearestNeighbour as KNN
from metric import *

def testKNN( ) :
    from tensorflow.keras.datasets import mnist

    def preprocess( dataset ) :
        # subsample to 14x14 (not doing this increases execution time by 4 times, since images are 4 times bigger)
        dataset = dataset.reshape(-1, 14, 2, 14, 2).max(axis=(2, 4))
        return dataset.reshape(len(dataset), -1)  # flatten 14x14 to 196

    from time import time
    start = time()
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = preprocess(train_images)
    test_images = preprocess(test_images)

    knn_classifier = KNN(10, (train_images, train_labels), ManhattanNorm)  # L1 norm : 81.34% accuracy, ~276s
    # knn_classifier = KNN(10, (train_images, train_labels), EuclideanNorm)   # L2 norm : 32.55% accuracy, ~282s
    correct = 0
    for image, label in zip(test_images, test_labels) :
        prediction = knn_classifier(image)
        correct += (label == prediction)

    print(f"Time elapsed : {time() - start}s")
    print(f"Accuracy = {correct}/{len(test_labels)} : {100 * correct / len(test_labels)}%")

if __name__ == '__main__' :
    testKNN()
