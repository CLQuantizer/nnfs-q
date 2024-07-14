import numpy as np
from classes import ActivationReLU, ActivationSoftmax, LayerDense, LossCategoricalCrossEntropy
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

x, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()
loss_function = LossCategoricalCrossEntropy()

dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:6])
loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print('acc', accuracy)