import numpy as np
import matplotlib.pyplot as plt

from dataset import load_svhn, random_split_train_val
from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient
from layers import FullyConnectedLayer, ReLULayer
from model import TwoLayerNet
from trainer import Trainer, Dataset
from optim import SGD, MomentumSGD
from metrics import multiclass_accuracy


def prepare_for_neural_network(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0

    # Subtract mean
    mean_image = np.mean(train_flat, axis=0)
    train_flat -= mean_image
    test_flat -= mean_image

    return train_flat, test_flat


train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)
train_X, test_X = prepare_for_neural_network(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val=1000)

# # TODO: Implement ReLULayer layer in layers.py
# # Note: you'll need to copy implementation of the gradient_check function from the previous assignment
#
#
# X = np.array([[1,-2,3],
#               [-1, 2, 0.1]
#               ])
#
# assert check_layer_gradient(ReLULayer(), X)
#
#
#
# # TODO: Implement FullyConnected layer forward and backward methods
# assert check_layer_gradient(FullyConnectedLayer(3, 4), X)
# # TODO: Implement storing gradients for W and B
# assert check_layer_param_gradient(FullyConnectedLayer(3, 4), X, 'W')
# assert check_layer_param_gradient(FullyConnectedLayer(3, 4), X, 'B')




# # TODO: In model.py, implement compute_loss_and_gradients function
# model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10, hidden_layer_size = 3, reg = 0)
# loss = model.compute_loss_and_gradients(train_X[:2], train_y[:2])
#
# # TODO Now implement backward pass and aggregate all of the params
# check_model_gradient(model, train_X[:2], train_y[:2])


model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10, hidden_layer_size = 100, reg = 1e1)
dataset = Dataset(train_X, train_y, val_X, val_y)
trainer = Trainer(model, dataset, SGD(), learning_rate = 1e-2)

# TODO Implement missing pieces in Trainer.fit function
# You should expect loss to go down every epoch, even if it's slow
loss_history, train_history, val_history = trainer.fit()