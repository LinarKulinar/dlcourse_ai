import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    probs = np.ones_like(predictions, dtype=float)
    if np.ndim(predictions) == 2:
        max_val = np.max(predictions, axis=1)
        predictions_new = predictions - max_val[:, np.newaxis]
        numerator = np.exp(predictions_new)
        denominator = np.sum(numerator, axis=1)
        probs = numerator / denominator[:, np.newaxis]
        return probs

    elif np.ndim(predictions) == 1:
        max_val = np.max(predictions)
        predictions_new = predictions - max_val
        denominator = np.sum(np.exp(predictions_new))
        probs = np.exp(predictions_new) / denominator
        return probs
    else:
        raise Exception("dim(predictions)!=1 or 2")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    batch_size = probs.shape[0]
    if np.ndim(probs) == 2:
        loss = 0
        for ind in range(len(target_index)):
             loss -= np.log(probs[ind, target_index[ind]])
        return loss[0]/batch_size
    elif np.ndim(probs) == 1:
        return -np.log(probs[target_index])/batch_size
    else:
        raise Exception("dim(predictions)!=1 or 2")

    # batch_size = probs.shape[0]
    #
    # return -np.log(probs[range(batch_size), target_index]).sum() / batch_size


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy

    probs = np.ones_like(predictions, dtype=float)

    batch_size = probs.shape[0]
    if np.ndim(predictions) == 2:

        max_val = np.max(predictions, axis=1)
        predictions_new = predictions - max_val[:, np.newaxis]
        numerator = np.exp(predictions_new)
        denominator = np.sum(numerator, axis=1)
        probs = numerator / denominator[:, np.newaxis]
        loss = 0
        for ind in range(len(target_index)):
            loss -= np.log(probs[ind, target_index[ind]])
            dprediction = probs
            dprediction[ind, target_index[ind]] = probs[ind, target_index[ind]] - 1

    elif np.ndim(predictions) == 1:
        max_val = np.max(predictions)
        predictions_new = predictions - max_val
        denominator = np.sum(np.exp(predictions_new))
        probs = np.exp(predictions_new) / denominator
        loss = -np.log(probs[target_index])
        dprediction = probs
        dprediction[target_index] = probs[target_index] - 1
    else:
        raise Exception("dim(predictions)!=1 or 2")

    dprediction /= batch_size
    loss /= batch_size

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # loss = reg_strength * np.linalg.norm(W, ord= # не подходит, тк тут ещё и модуль берется
    norm2 = 0
    for i in W:
        for obj in i:
            norm2 += obj ** 2
    loss = reg_strength * norm2
    grad = reg_strength * 2 * W
    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            for ind in batches_indices:
                # Compute loss and gradients
                loss1, dW1 = linear_softmax(X[ind], self.W, y[ind])
                loss2, dW2 = l2_regularization(self.W, reg)
                loss = loss1 + loss2
                grad = dW1+dW2
                # Apply gradient to weights using learning rate
                self.W = self.W - learning_rate * grad
                # Don't forget to add both cross-entropy loss
                # and regularization!
                loss_history.append(loss)
            print("\tEpoch %i, loss: %f" % (epoch, loss))


        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        probs = softmax(np.dot(X, self.W))
        if np.ndim(probs) == 1:
            y_pred = probs.argmax()
        if np.ndim(probs) == 2:
            y_pred = probs.argmax(axis=1)
        return y_pred
