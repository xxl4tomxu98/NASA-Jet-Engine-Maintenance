import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Masking
from tensorflow.keras.optimizers.legacy import RMSprop
from sklearn.preprocessing import normalize
from wtte import wtte as wtte


"""
    Load and parse engine data files into:
       - an (engine/day, observed history, sensor readings) x tensor,
         where observed history is 100 days, zero-padded for days
         that don't have a full 100 days of observed history
         (e.g., first observed day for an engine)
       - an (engine/day, 2) tensor containing time-to-event and 1
         (since all engines failed)
    There are probably MUCH better ways of doing this, but I don't use
    Numpy that much, and the data parsing isn't the point of this demo anyway.
"""


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


def build_data(engine, time, x, max_time, is_test):
    # y[0] will be days remaining, y[1] will be event indicator,
    # always 1 for this data
    out_y = np.empty((0, 2), dtype=np.float32)
    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)
    for i in range(100):
        # print("Engine: " + str(i))
        # When did the engine fail? (Last day + 1 for train data,
        # irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1
        if is_test:
            start = max_engine_time - 1
        else:
            start = 0
        this_x = np.empty((0, max_time, 24), dtype=np.float32)
        for j in range(start, max_engine_time):
            engine_x = x[engine == i]
            out_y = np.append(out_y,
                              np.array((max_engine_time - j, 1), ndmin=2),
                              axis=0)
            xtemp = np.zeros((1, max_time, 24))
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = \
                engine_x[max(0, j-max_time+1):j+1, :]
            this_x = np.concatenate((this_x, xtemp))
        out_x = np.concatenate((out_x, this_x))
    return out_x, out_y


train_x, train_y = build_data(train[:, 0], train[:, 1],
                              train[:, 2:26], max_time, False)
test_x = build_data(test_x[:, 0], test_x[:, 1],
                    test_x[:, 2:26], max_time, True)[0]

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
model.compile(loss=wtte.loglik_discrete, optimizer=RMSprop(learning_rate=.001))
# Fit!
model.fit(train_x, train_y, epochs=5, batch_size=2000,
          verbose=2, validation_data=(test_x, test_y))

# Make predictions and put alongside the real TTE and event indicator
test_predict = model.predict(test_x)
test_predict = np.resize(test_predict, (100, 2))
test_result = np.concatenate((test_y, test_predict), axis=1)

# TTE, Event Indicator, Alpha, Beta
print(test_result)
