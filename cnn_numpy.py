# cnn_numpy.py
# Written by David Stein (david@djstein.com). See https://www.djstein.com/cnn-numpy/ for more info.
# Source: https://github.com/neuron-whisperer/cnn-numpy

# This code is an adaptation of Andrew Ng's convolutional neural network code:
#   https://www.coursera.org/learn/convolutional-neural-networks/programming/qO8ng/convolutional-model-step-by-step

import math, numpy as np, random

class ConvLayer:

  def __init__(self, num_filters, filter_width, stride = 1, padding = 0):
    self.fw = filter_width; self.n_f = num_filters; self.s = stride; self.p = padding
    self.W = None; self.b = None

  def forward_propagate(self, input):
    self.input = np.array(input)
    if self.p > 0:                                  # pad input
      shape = ((0, 0), (self.p, self.p), (self.p, self.p), (0, 0))
      self.input = np.pad(input, shape, mode='constant', constant_values = (0, 0))
    self.W = np.random.random((self.fw, self.fw, self.input.shape[3], self.n_f)) * 0.01
    self.b = np.random.random((1, 1, 1, self.n_f)) * 0.01
    self.n_m = self.input.shape[0]                                        # number of inputs
    self.ih = self.input.shape[1]; self.iw = self.input.shape[2]          # input height and width
    self.oh = math.floor((self.ih - self.fw + 2 * self.p) / self.s) + 1   # output height
    self.ow = math.floor((self.iw - self.fw + 2 * self.p) / self.s) + 1   # output width
    self.Z = np.zeros((self.n_m, self.oh, self.ow, self.n_f))
    for i in range(self.n_m):                       # iterate over inputs
      for h in range(self.oh):                      # iterate over output height
        ih1 = h * self.s; ih2 = ih1 + self.fw       # calculate input window height coordinates
        for w in range(self.ow):                    # iterate over output width
          iw1 = w * self.s; iw2 = iw1 + self.fw     # calculate input window width coordinates
          for f in range(self.n_f):                 # iterate over filters
            self.Z[i, h, w, f] = np.sum(self.input[i, ih1:ih2, iw1:iw2, :] * self.W[:, :, :, f])
            self.Z += self.b[:, :, :, f]            # calculate output
    return self.Z

  def backpropagate(self, dZ, learning_rate):
    dA_prev = np.zeros((self.n_m, self.ih, self.iw, self.n_f))
    dW = np.zeros(self.W.shape); db = np.zeros(self.b.shape)
    for i in range(self.n_m):                       # iterate over inputs
      for h in range(self.oh):                      # iterate over output width
        ih1 = h * self.s; ih2 = ih1 + self.fw       # calculate input window height coordinates
        for w in range(self.ow):                    # iterate over output width
          iw1 = w * self.s; iw2 = iw1 + self.fw     # calculate input window width coordinates
          for f in range(self.n_f):                 # iterate over filters
            dA_prev[i, ih1:ih2, iw1:iw2, :] += self.W[:, :, :, f] * dZ[i, h, w, f]
            dW[:, :, :, f] += self.input[i, ih1:ih2, iw1:iw2, :] * dZ[i, h, w, f]
            db[:, :, :, f] += dZ[i, h, w, f]
    self.W -= dW * learning_rate; self.b -= db * learning_rate
    if self.p > 0:                                  # remove padding
      dA_prev = dA_prev[:, self.p:-self.p, self.p:-self.p, :]
    return dA_prev

class PoolLayer:

  def __init__(self, filter_width, stride = 1):
    self.fw = filter_width; self.s = stride
  
  def forward_propagate(self, input):
    self.input = input
    self.n_m = self.input.shape[0]                                  # number of inputs
    self.ih = self.input.shape[1]; self.iw = self.input.shape[2]    # input height and width
    self.oh = math.floor((self.ih - self.fw) / self.s) + 1          # output width
    self.ow = math.floor((self.iw - self.fw) / self.s) + 1          # output height
    self.n_f = self.input.shape[3]                                  # output channels (same as input channels)
    self.Z = np.zeros((self.n_m, self.oh, self.ow, self.n_f))
    for i in range(self.n_m):                       # iterate over inputs
      for h in range(self.oh):                      # iterate over output height
        ih1 = h * self.s; ih2 = ih1 + self.fw       # calculate input window height coordinates
        for w in range(self.ow):                    # iterate over output width
          iw1 = w * self.s; iw2 = iw1 + self.fw     # calculate input window width coordinates
          for f in range(self.n_f):                 # iterate over output channels
            self.Z[i, h, w, f] = self.pool(self.input[i, ih1:ih2, iw1:iw2, f])
    return self.Z

  def backpropagate(self, dZ, learning_rate):
    dA_prev = np.zeros((self.n_m, self.ih, self.iw, self.n_f))
    for i in range(self.n_m):                       # iterate over input images
      for h in range(self.oh):                      # iterate over output height
        ih1 = h * self.s; ih2 = ih1 + self.fw       # calculate input window height coordinates
        for w in range(self.ow):                    # iterate over output width
          iw1 = w * self.s; iw2 = iw1 + self.fw     # calculate input window width coordinates
          for f in range(self.n_f):                 # iterate over output channels
            slice = self.input[i, ih1:ih2, iw1:iw2, f]
            dA_prev[i, ih1:ih2, iw1:iw2, f] += self.gradient(slice, dZ[i, h, w, f])
    return dA_prev
    
class PoolLayer_Max(PoolLayer):

  def __init__(self, filter_width, stride = 1):
    self.pool = np.max
    self.gradient = lambda slice, dZ: dZ * (slice == np.max(slice))
    super().__init__(filter_width, stride)
  
class PoolLayer_Avg(PoolLayer):

  def __init__(self, filter_width, stride = 1):
    self.pool = np.mean
    self.gradient = lambda slice, dZ: np.ones((self.fw, self.fw)) * dZ / (self.fw ** 2)
    super().__init__(filter_width, stride)

class FlatLayer:

  def forward_propagate(self, input):
    self.input_shape = input.shape
    return np.reshape(input, (input.shape[0], int(input.size / input.shape[0])))

  def backpropagate(self, dZ, learning_rate):
    return np.reshape(dZ, self.input_shape)

class FCLayer:

  def __init__(self, num_neurons):
    self.num_neurons = num_neurons; self.W = None

  def forward_propagate(self, input):
    if self.W is None:
      self.W = np.random.random((self.num_neurons, input.shape[1] + 1)) * 0.0001
    self.input = np.hstack([input, np.ones((input.shape[0], 1))])  # add bias inputs
    self.Z = np.dot(self.input, self.W.transpose())
    return self.activate(self.Z)

  def backpropagate(self, dA, learning_rate):
    dZ = self.gradient(dA, self.Z)
    dW = np.dot(self.input.transpose(), dZ).transpose() / dA.shape[0]
    dA_prev = np.dot(dZ, self.W)
    dA_prev = np.delete(dA_prev, dA_prev.shape[1] - 1, 1)          # remove bias inputs
    self.W = self.W - learning_rate * dW
    return dA_prev

class FCLayer_ReLU(FCLayer):

  def __init__(self, num_neurons):
    self.activate = lambda Z: np.maximum(0.0, Z)
    self.gradient = lambda dA, Z: dA * (Z > 0.0)
    super().__init__(num_neurons)

class FCLayer_Sigmoid(FCLayer):

  def __init__(self, num_neurons):
    self.activate = lambda Z: 1.0 / (1.0 + np.exp(-Z))
    self.gradient = lambda dA, Z: dA / (1.0 + np.exp(-Z)) * (1.0 - (1.0 / (1.0 + np.exp(-Z))))
    super().__init__(num_neurons)

class FCLayer_Softmax(FCLayer):

    def __init__(self, num_neurons):
        self.activate = lambda Z: np.exp(1.0 / (1.0 + np.exp(-Z))) / np.expand_dims(np.sum(np.exp(1.0 / (1.0 + np.exp(-Z))), axis=1), 1)
        self.gradient = lambda dA, Z: dA / (1.0 + np.exp(-Z)) * (1.0 - (1.0 / (1.0 + np.exp(-Z))))
        super().__init__(num_neurons)

class Network:

  def __init__(self, layers = []):
    self.layers = layers
  
  def predict(self, X):
    A = np.array(X)
    for i in range(len(self.layers)):
      A = self.layers[i].forward_propagate(A)
    A = np.clip(A, 1e-15, None)                   # clip to avoid log(0) in CCE
    A += np.random.random(A.shape) * 0.00001      # small amount of noise to break ties
    return A
  
  def evaluate(self, X, Y):
    A = self.predict(X); Y = np.array(Y)
    cce = -np.sum(Y * np.log(A)) / A.shape[0]     # categorical cross-entropy
    B = np.array(list(1.0 * (A[i] == np.max(A[i])) for i in range(A.shape[0])))
    ce = np.sum(np.abs(B - Y)) / len(Y) / 2.0     # class error
    return (A, cce, ce)

  def train(self, X, Y, learning_rate):
    A, cce, ce = self.evaluate(X, Y)
    dA = A - Y
    for i in reversed(range(len(self.layers))):
      dA = self.layers[i].backpropagate(dA, learning_rate)
    return (np.copy(self.layers), cce, ce)
