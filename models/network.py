import tensorflow as tf
import numpy as np
import os
import constants
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

train_data_path = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data_path = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'

BATCH_SIZE = 32
EPOCHS = 35
loss_funcs = ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error']

def load_train_data(path, num):
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


def load_test_data():
    X = load_train_data(test_data_path, constants.num_of_test_data)
    read_test_labels = open(test_labels, 'r')
    y_test = read_test_labels.read().split('\n')
    y_test = y_test[:-1]
    newY = []
    for y in y_test:
        newY.append([int(char) for char in y])
    y_test = newY
    read_test_labels.close()
    return X, y_test


# efficient net https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0
model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(33, 33, 1),
    pooling=None,
    classes=180,
    classifier_activation=None
)

for loss_func in loss_funcs:
    model.compile(
        optimizer='adagrad',
        loss=loss_func,
        metrics=[tf.keras.metrics.MeanSquaredError()],
        loss_weights=None, weighted_metrics=None,
    )

    # training
    X = load_train_data(train_data_path, constants.num_of_train_data)  # numpy array of input QR codes
    read_train_labels = open(train_labels, 'r')
    Y = read_train_labels.read().split('\n')  # the corresponding appended query string
    Y = Y[:-1]  # remove last element because of trailing new line
    newY = []
    for y in Y:
        newY.append([int(char) for char in y])
    Y = newY
    read_train_labels.close()
    print('Training the model')
    history = model.fit(x=X, y=tf.convert_to_tensor(Y, dtype=tf.int32), epochs=EPOCHS, validation_split=.2)
    model.save_weights('my_qr_network')

    # testing
    print('************************ Evaluate on test data')
    X, y_test = load_test_data()
    results = model.evaluate(x=X, y=tf.convert_to_tensor(y_test, dtype=tf.int32))
    print('test loss, test acc:', results)
    results_file = open('results_network.txt', 'a')
    results_file.write(loss_func + '\ntest loss: ' + str(results[0]) + '   test acc: ' + str(results[1]) + '\n\n')
    results_file.close()

    predictions = model.predict(x=X)
    print(predictions[0])
    print(y_test[0])