import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    # Мой неоптимальный код:
    # norm2 = 0
    # for i in W:
    #     for obj in i:
    #         norm2 += obj ** 2
    # loss = reg_strength * norm2
    # grad = reg_strength * 2 * W

    loss = (W ** 2).sum() * reg_strength
    grad = W * 2 * reg_strength

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
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


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(self.value)  # reset_grad

    def reset_grad(self):
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        self.relu = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass

        self.relu = np.maximum(X, 0)
        return self.relu

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops

        return d_out * np.sign(self.relu)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return np.dot(X, self.W.value)+self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to
           input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
