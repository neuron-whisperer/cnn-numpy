# mnist.py
# Written by David Stein (david@djstein.com). See https://www.djstein.com/cnn-numpy/ for more info.
# Source: https://github.com/neuron-whisperer/cnn-numpy

import gzip, math, numpy as np, os, random, requests, sys, time, warnings

def load_mnist_database():
  # read MNIST database according to format (http://yann.lecun.com/exdb/mnist/)
  path = os.path.dirname(os.path.realpath(__file__)); data_set = []
  for name in ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
    full_name = f'{path}/{name}'
    if not os.path.exists(full_name):
      r = requests.get(f'http://yann.lecun.com/exdb/mnist/{name}')
      with open(full_name, 'wb') as file:
        file.write(r.content)
    f = gzip.open(full_name)
    if 'images' in name:
      header = f.read(16)
      num_images = int.from_bytes(header[4:8], byteorder='big')
      image_size = int.from_bytes(header[8:12], byteorder='big')
      buf = f.read(image_size * image_size * num_images)
      data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
      data_set.append(data.reshape(num_images, image_size, image_size, 1))
    else:
      header = f.read(8)
      num_labels = int.from_bytes(header[4:8], byteorder='big')
      buf = f.read(image_size * image_size * num_images)
      data_set.append(np.frombuffer(buf, dtype=np.uint8))
  x = np.concatenate((data_set[0], data_set[1])); y = np.concatenate((data_set[2], data_set[3]))
  num_classes = np.max(y) + 1; y = np.eye(num_classes)[y]    # one-hot-encoded labels
  return (x, y)

def split_database(x, y, ratio):
  indices = list(range(len(x))); random.shuffle(indices)
  split = int(len(x) * ratio)
  train_x = list(x[i] for i in indices[:split])
  test_x = list(x[i] for i in indices[split:])
  train_y = list(y[i] for i in indices[:split])
  test_y = list(y[i] for i in indices[split:])
  return ((train_x, train_y), (test_x, test_y))

def train_mnist(net, learning_rate, num_epochs, mini_batches, split):
  x, y = load_mnist_database(); history = []
  training_set, test_set = split_database(x, y, split)
  start_time = time.time()
  for e in range(num_epochs):
    for b in range(mini_batches):
      start = int(len(training_set[0]) * b / mini_batches)
      stop = int(len(training_set[0]) * (b + 1) / mini_batches)
      mb_x = training_set[0][start:stop]; mb_y = training_set[1][start:stop]
      history.append(net.train(mb_x, mb_y, learning_rate))
      t = int(time.time() - start_time); t_str = '%2dm %2ds' % (int(t / 60), t % 60)
      _, cce, ce = history[-1]
      print(f'\rTime: {t_str}  Epoch: {e+1:4}, {b+1:4}/{mini_batches}  Mini-Batch Size: {len(mb_x)}  CCE: {cce:0.4f}  CE: {ce:0.4f}', end='')
    print('')
  cce, ce = net.run(test_set[0], test_set[1])
  print(f'Test set: CCE: {cce:0.3f}  CE: {ce:0.3f}')

if __name__ == '__main__':

  use_cnn = (len(sys.argv) > 1 and sys.argv[1].lower() == 'cnn')
  use_flat = (len(sys.argv) > 1 and sys.argv[1].lower() == 'flat')
  if not (use_cnn or use_flat):
    print('Syntax: python3 mnist.py (cnn | flat) [sg]'); sys.exit(1)
  naive = (len(sys.argv) > 2 and sys.argv[2].lower() == 'naive')
  if naive:
    from cnn_numpy import *       # naive implementation
  else:
    from cnn_numpy_sg import *    # stride groups implementation
    
  with warnings.catch_warnings():
    np.set_printoptions(threshold=sys.maxsize)
    warnings.simplefilter("ignore")     # disable numpy overflow warnings
    net = [FlatLayer(), FCLayer_ReLU(100), FCLayer_Softmax(10)]
    learning_rate = 0.01; num_epochs = 100; mini_batches = 10
    if use_cnn:
      net[0:0] = [ConvLayer(32, 2), PoolLayer_Max(3, 3)]
      learning_rate = 0.001; num_epochs = 3; mini_batches = 6550 if naive else 500
    train_mnist(Network(net), learning_rate, num_epochs, mini_batches, split = 0.95)
