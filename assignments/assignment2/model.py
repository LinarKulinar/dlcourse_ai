import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param in self.params().values():
            param.reset_grad()

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        result_answer = X.copy()
        for layer in self.layers:
            result_answer = layer.forward(result_answer)
        loss, dprediction = softmax_with_cross_entropy(result_answer, y)

        dW = dprediction
        for layer in reversed(self.layers):
            dW = layer.backward(dW)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        for param in self.params().values():
            loss_reg, dW2 = l2_regularization(param.value, self.reg)
            loss += loss_reg
            param.grad += dW2  # накапливаем градиенты

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        #pred = np.zeros(X.shape[0], np.int)

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)
        return np.argmax(pred, axis=1)  # выдаем номер того, у кого pred наибольший в данном sample

    def params(self):
        # TODO Implement aggregating all of the params
        result = {}
        for i, lay in enumerate(self.layers):  # идем по всем слоям
            for param in lay.params():  # идем по всем параметрам конкретного слоя
                result[param + str(i)] = lay.params()[param]  # кладем в словарь
        return result
