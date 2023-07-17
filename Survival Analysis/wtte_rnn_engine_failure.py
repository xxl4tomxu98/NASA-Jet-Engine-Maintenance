from wtte import wtte as wtte  # noqa
from wtte import pipelines as pipelines
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Masking
from tensorflow.keras.optimizers.legacy import RMSprop
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, '..')


def load_file(name):
    with open(name, 'r') as file:
        return np.loadtxt(file, delimiter=',')


path = "/Users/tomxu/Documents/NASA-Jet-Engine-Maintenance/Data/wtte_keras/"
np.set_printoptions(suppress=True, threshold=10000)
train = load_file(path + 'train.csv')
test_x = load_file(path + 'test_x.csv')
test_y = load_file(path + 'test_y.csv')
# Combine the X values to normalize them, then split them back out
all_x = np.concatenate((train[:, 2:26], test_x[:, 2:26]))
all_x = normalize(all_x, axis=0)
train[:, 2:26] = all_x[0:train.shape[0], :]
test_x[:, 2:26] = all_x[train.shape[0]:, :]
# Make engine numbers and days zero-indexed, for everybody's sanity
train[:, 0:2] -= 1
test_x[:, 0:2] -= 1
# Configurable observation look-back period for each engine/day
max_time = 100
mask_value = -99

train_x, train_y = pipelines.build_data(train[:, 0], train[:, 1],
                                        train[:, 2:26], max_time,
                                        False, mask_value)
test_x = pipelines.build_data(test_x[:, 0], test_x[:, 1],
                              test_x[:, 2:26], max_time,
                              True, mask_value)[0]

train_u = np.zeros((100, 1), dtype=np.float32)
train_u += 1
test_y = np.append(np.reshape(test_y, (100, 1)), train_u, axis=1)
train_y = tf.cast(train_y, tf.float32)
test_y = tf.cast(test_y, tf.float32)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
print("train_x: ", train_x)
print("train_y: ", train_y)
"""
    Here's the rest of the meat of the demo... actually fitting and training
    the model. We'll also make some test predictions so we can evaluate model
    performance.
"""
# Start building our model
model = Sequential()
# Mask parts of the lookback period that are all zeros (i.e., unobserved) so
# they don't skew the model
model.add(Masking(mask_value=0., input_shape=(max_time, 24)))
# LSTM is just a common type of RNN.
model.add(LSTM(20, input_dim=24))
# We need 2 neurons to output Alpha and Beta parameters for our Weibull
model.add(Dense(2))
# Apply the custom activation function mentioned above
model.add(Activation(wtte.activate))
# Use the discrete log-likelihood for Weibull survival as loss function
loss = wtte.loss(kind='discrete', reduce_loss=False).loss_function
model.compile(loss=loss, optimizer=RMSprop(learning_rate=.001))
# Fit!
model.fit(train_x, train_y, epochs=5, batch_size=2000,
          verbose=2, validation_data=(test_x, test_y))

# Make predictions and put alongside the real TTE and event indicator
test_predict = model.predict(test_x)
test_predict = np.resize(test_predict, (100, 2))
test_result = np.concatenate((test_y, test_predict), axis=1)

# TTE, Event Indicator, Alpha, Beta
print(test_result)
