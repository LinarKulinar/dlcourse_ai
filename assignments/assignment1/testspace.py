import numpy as np
import matplotlib.pyplot as plt

from dataset import load_svhn, random_split_train_val
from gradient_check import check_gradient
from metrics import multiclass_accuracy
import linear_classifer


def prepare_for_linear_classifier(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0

    # Subtract mean (вычитам среднее)
    mean_image = np.mean(train_flat, axis=0)
    train_flat -= mean_image
    test_flat -= mean_image

    # Add another channel with ones as a bias term
    train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0], 1))])
    test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0], 1))])
    return train_flat_with_ones, test_flat_with_ones


train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)
train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val=1000)

#
# # TODO: Implement gradient check function
# def sqr(x):
#     return x * x, 2 * x
#
#
# check_gradient(sqr, np.array([3.0]))
#
#
# def array_sum(x):
#     # assert x.shape == (2,), x.shape
#     return np.sum(x), np.ones_like(x)
#
#
# check_gradient(array_sum, np.array([3.0, 2.0]))
#
#
# def array_2d_sum(x):
#     # assert x.shape == (2,2)
#     return np.sum(x), np.ones_like(x)
#
#
# check_gradient(array_2d_sum, np.array([[3.0, 2.0], [1.0, 0.0]]))
#
#
# # TODO Implement softmax and cross-entropy for single sample
# probs = linear_classifer.softmax(np.array([-10, 0, 10]))
#
# # Make sure it works for big numbers too!
# probs = linear_classifer.softmax(np.array([1000, 0, 0]))
# assert np.isclose(probs[0], 1.0)
#
# probs = linear_classifer.softmax(np.array([[-10, 0, 10],[10, 0, -10]]))
# print(probs)
#
#
#
# probs = linear_classifer.softmax(np.array([[-5, 0, 5],[5, 0, -5]]))
# print("probs = ", probs)
# h = linear_classifer.cross_entropy_loss(probs, np.array([[0],[1]]))
# print("h = ", h)
#
#
#
#
# loss, grad = linear_classifer.softmax_with_cross_entropy(np.array([1, 0, 0]), 1)
# print(loss)
# print(grad)
# check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, 1), np.array([1, 0, 0], np.float))


# TODO Extend combined function so it can receive a 2d array with batch of samples

# Test batch_size = 1
batch_size = 1
predictions = np.zeros((batch_size, 3))
target_index = np.ones(batch_size, np.int)
check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, target_index), predictions)

# Test batch_size = 3
batch_size = 3
predictions = np.zeros((batch_size, 3))
target_index = np.ones(batch_size, np.int)
check_gradient(lambda x: linear_classifer.softmax_with_cross_entropy(x, target_index), predictions)