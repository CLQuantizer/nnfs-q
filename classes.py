import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        self.inputs = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


class ActivationReLU:
    def __init__(self):
        self.dinputs = None
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    # there ain't no weights or biases in ReLU
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0


class ActivationSoftmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    #
    # def backward(self, dvalues):
    #



class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples_length = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = None
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples_length), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        # correct_confidences = y_pred_clipped[range(samples_length), y_true]
        # Example: if y_pred_clipped is:
        # [[0.1, 0.6, 0.2, 0.1],
        #  [0.7, 0.1, 0.1, 0.1],
        #  [0.2, 0.2, 0.2, 0.4]]
        # and y_true is [1, 0, 3]
        # Then correct_confidences will be [0.6, 0.7, 0.4]
        return negative_log_likelihoods
