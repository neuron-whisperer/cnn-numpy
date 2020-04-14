# cnn_numpy_sg.py
# Written by David Stein (david@djstein.com). See https://www.djstein.com/cnn-numpy/ for more info.
# Source: https://github.com/neuron-whisperer/cnn-numpy

# This code improves upon cnn_numpy.py by implementing ConvLayer and PoolLayers using stride groups.

import math, numpy as np, random

class ConvLayer:

  def __init__(self, num_filters, filter_width, stride = 1, padding = 0):
    self.fw = filter_width; self.n_f = num_filters; self.s = stride; self.p = padding
    self.W = None; self.b = None

  def forward_propagate(self, input):
    input = np.array(input)
    if self.p > 0:                                  # pad input
      shape = ((0, 0), (self.p, self.p), (self.p, self.p), (0, 0))
      input = np.pad(input, shape, mode='constant', constant_values = (0, 0))
    im, ih, iw, id = input.shape; s = self.s; fw = self.fw; f = self.n_f
    self.input_shape = input.shape
    if self.W is None:
      self.W = np.random.random((self.fw, self.fw, id, self.n_f)) * 0.1
      self.b = np.random.random((1, 1, 1, self.n_f)) * 0.01
    self.n_rows = math.ceil(min(fw, ih - fw + 1) / s)
    self.n_cols = math.ceil(min(fw, iw - fw + 1) / s)
    z_h = int(((ih - fw) / s) + 1); z_w = int(((iw - fw) / s) + 1)
    self.Z = np.empty((im, z_h, z_w, f)); self.input_blocks = []
    for t in range(self.n_rows): 
      self.input_blocks.append([])
      b = ih - (ih - t) % fw
      cols = np.empty((im, int((b - t) / fw), z_w, f))
      for i in range(self.n_cols):
        l = i * s; r = iw - (iw - l) % fw
        block = input[:, t:b, l:r, :]
        block = np.array(np.split(block, (r - l) / fw, 2))
        block = np.array(np.split(block, (b - t) / fw, 2))
        block = np.moveaxis(block, 2, 0)
        block = np.expand_dims(block, 6)
        self.input_blocks[t].append(block)
        block = block * self.W
        block = np.sum(block, 5)
        block = np.sum(block, 4)
        block = np.sum(block, 3)
        cols[:, :, i::self.n_cols, :] = block
      self.Z[:, t * s ::self.n_rows, :, :] = cols
    self.Z += self.b
    self.A = np.abs(self.Z)                           # ReLU activation
    return self.A

  def backpropagate(self, dZ, learning_rate):
    im, ih, iw, id = self.input_shape; s = self.s; fw = self.fw; f = self.n_f
    n_rows = self.n_rows; n_cols = self.n_cols
    dA_prev = np.zeros((im, ih, iw, id))
    dW = np.zeros(self.W.shape); db = np.zeros(self.b.shape)
    for t in range(n_rows):
      row = dZ[:, t::n_rows, :, :]
      for l in range(n_cols):
        b = (ih - t * s) % fw; r = (iw - l * s) % fw  # region of input and dZ for this block
        block = row[:, :, l * s::n_cols, :]           # block = corresponding region of dA
        block = np.expand_dims(block, 3)              # axis for channels
        block = np.expand_dims(block, 3)              # axis for rows
        block = np.expand_dims(block, 3)              # axis for columns
        dW_block = block * self.input_blocks[t][l]
        dW_block = np.sum(dW_block, 2)
        dW_block = np.sum(dW_block, 1)
        dW_block = np.sum(dW_block, 0)
        dW += dW_block
        db_block = np.sum(dW_block, 2, keepdims=True)
        db_block = np.sum(db_block, 1, keepdims=True)
        db_block = np.sum(db_block, 0, keepdims=True)
        db += db_block
        dA_prev_block = block * self.W
        dA_prev_block = np.sum(dA_prev_block, 6)
        dA_prev_block = np.reshape(dA_prev_block, (im, ih - b - t, iw - r - l, id))
        dA_prev[:, t:ih - b, l:iw - r, :] += dA_prev_block
    self.W -= dW * learning_rate; self.b -= db * learning_rate
    if self.p > 0:                                   # remove padding
      dA_prev = dA_prev[:, self.p:-self.p, self.p:-self.p, :]
    return dA_prev

class PoolLayer:

  def __init__(self, filter_width, stride = 1):
    self.fw = filter_width; self.s = stride
  
  def forward_propagate(self, input):
    im, ih, iw, id = input.shape; fw = self.fw; s = self.s
    self.n_rows = math.ceil(min(fw, ih - fw + 1) / s)
    self.n_cols = math.ceil(min(fw, iw - fw + 1) / s)
    z_h = int(((ih - fw) / s) + 1); z_w = int(((iw - fw) / s) + 1)
    self.Z = np.empty((im, z_h, z_w, id)); self.input = input
    for t in range(self.n_rows): 
      b = ih - (ih - t) % fw
      Z_cols = np.empty((im, int((b - t) / fw), z_w, id))
      for i in range(self.n_cols):
        l = i * s; r = iw - (iw - l) % fw
        block = input[:, t:b, l:r, :]
        block = np.array(np.split(block, (r - l) / fw, 2))
        block = np.array(np.split(block, (b - t) / fw, 2))
        block = self.pool(block, 4)
        block = self.pool(block, 3)
        block = np.moveaxis(block, 0, 2)
        block = np.moveaxis(block, 0, 2)
        Z_cols[:, :, i::self.n_cols, :] = block
      self.Z[:, t * s ::self.n_rows, :, :] = Z_cols
    return self.Z

  def assemble_block(self, block, t, b, l, r):
    ih = self.input.shape[1]; iw = self.input.shape[2]
    block = np.repeat(block, self.fw ** 2, 2)
    block = np.array(np.split(block, block.shape[2] / self.fw, 2))
    block = np.moveaxis(block, 0, 2)
    block = np.array(np.split(block, block.shape[2] / self.fw, 2))
    block = np.moveaxis(block, 0, 3)
    return np.reshape(block, (self.input.shape[0], ih - t - b, iw - l - r, self.input.shape[3]))

class PoolLayer_Max(PoolLayer):

  def __init__(self, filter_width, stride = 1):
    self.pool = np.max
    super().__init__(filter_width, stride)
  
  def backpropagate(self, dZ, learning_rate):
    im, ih, iw, id = self.input.shape
    fw = self.fw; s = self.s; n_rows = self.n_rows; n_cols = self.n_cols
    dA_prev = np.zeros(self.input.shape)
  
    for t in range(n_rows):
      mask_row = self.Z[:, t::n_rows, :, :]
      row = dZ[:, t::self.n_rows, :, :]
      for l in range(self.n_cols):
        b = (ih - t * s) % fw; r = (iw - l * s) % fw
        mask = mask_row[:, :, l * s::n_cols, :]
        mask = self.assemble_block(mask, t, b, l, r)
        block = row[:, :, l * s::n_cols, :]
        block = self.assemble_block(block, t, b, l, r)
        mask = (self.input[:, t:ih - b, l:iw - r, :] == mask)
        dA_prev[:, t:ih - b, l:iw - r, :] += block * mask
    return dA_prev

class PoolLayer_Avg(PoolLayer):

  def __init__(self, filter_width, stride = 1):
    self.pool = np.mean
    super().__init__(filter_width, stride)

  def backpropagate(self, dZ, learning_rate):
    im, ih, iw, id = self.input.shape
    fw = self.fw; s = self.s; n_rows = self.n_rows; n_cols = self.n_cols
    dA_prev = np.zeros(self.input.shape)
  
    for t in range(n_rows):
      row = dZ[:, t::n_rows, :, :]
      for l in range(n_cols):
        b = (ih - t * s) % fw; r = (iw - l * s) % fw
        block = row[:, :, l * s::n_cols, :]
        block = self.assemble_block(block, t, b, l, r)
        dA_prev[:, t:ih - b, l:iw - r, :] += block / (fw ** 2)
    return dA_prev

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
