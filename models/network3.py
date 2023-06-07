import tensorflow as tf
import numpy as np
import os
import constants
from tensorflow.keras import losses, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

train_data_path = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data_path = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'

BATCH_SIZE = 32
EPOCHS = 50


def load_my_data(path, num):
    """
    this method is for loading the qr code matrices that will be used as X input
    :param path: the path to your data
    :param num: the number of data
    :return: nparray of your data in the shape (num, 33, 33)
    """
    file = open(path, 'r')
    data = np.loadtxt(file, delimiter=',', ndmin=2).reshape(num, 33, 33)
    file.close()
    return data


@tf.function
def round_output(x):
    return tf.round(x)


def _create_model(opt='adam', ha='sigmoid', oa='sigmoid') -> models.Sequential:
    hidden_activation = ha
    output_activation = round_output
    optimizer = opt
    model = models.Sequential()
    # first layer
    model.add(Dense(180, activation=hidden_activation, input_shape=(33, 33, 1)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # second layer
    model.add(Dense(360, activation=hidden_activation))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # third layer
    model.add(Dense(360, activation=hidden_activation))
    # output layer
    model.add(Flatten())
    model.add(Dense(180, activation=output_activation))
    # compile
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


model = _create_model(opt='adagrad')

# training
X = load_my_data(train_data_path, constants.num_of_train_data)  # numpy array of input QR codes
read_train_labels = open(train_labels, 'r')
Y = read_train_labels.read().split('\n')  # the corresponding appended query string
Y = Y[:-1]  # remove last element because of trailing new line
newY = []
for y in Y:
    newY.append([int(char) for char in y])
Y = newY
read_train_labels.close()
print('Training the model')
history = model.fit(x=X, y=tf.constant(Y, dtype=tf.int32), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=.2)
model.save('my_qr_network3')

# testing
X = load_my_data(test_data_path, constants.num_of_test_data)
read_test_labels = open(test_labels, 'r')
y_test = read_test_labels.read().split('\n')
y_test = y_test[:-1]
newY = []
for y in y_test:
    newY.append([int(char) for char in y])
y_test = newY
read_test_labels.close()
print('Evaluate on test data')
results = model.evaluate(x=X, y=tf.convert_to_tensor(y_test, dtype=tf.int32))
print('test loss, test acc:', results)
results_file = open('results_network3.txt', 'a+')
results_file.write('test loss: ' + str(results[0]) + ' test acc: ' + str(results[1]))
results_file.close()
