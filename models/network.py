import tensorflow as tf
import numpy as np
import os
import constants
from plot_graph import plot_performance
from mappings import output_mapping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

train_data_path = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data_path = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'

BATCH_SIZE = 64
EPOCHS = 1
# reverse the char to binary dict
binary_to_char = {value: key for key, value in output_mapping.items()}


def binary_to_string(binary):
    """
    :param binary: a list of 0 and 1
    :return: the corresponding alphanumeric string
    """
    string = ''
    for x in range(0, len(binary)-6, 6):
        b = binary[x:x+6]
        temp = ''.join(str(num) for num in b)
        string += binary_to_char[temp]
    return string


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
    classifier_activation='sigmoid'  # TODO sigmoid
)

# loss_funcs = ['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error']
# opt_funcs = ['adagrad', 'adamax', 'adam', 'sgd', 'adadelta']
opt_funcs = ['sgd']
loss_funcs = ['mean_squared_error']
for loss_func in loss_funcs:
    for opt_func in opt_funcs:
        model.compile(
            optimizer=opt_func,
            loss=loss_func,
            metrics=[tf.keras.metrics.MeanSquaredError()],
            loss_weights=None, weighted_metrics=None,
        )
        model.load_weights('my_qr_network')

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
        plot_performance(history, title='plot_1_' + loss_func + '_' + opt_func)

        # testing
        print('************************ Evaluate on test data')
        X, y_test = load_test_data()
        results = model.evaluate(x=X, y=tf.convert_to_tensor(y_test, dtype=tf.int32))
        print('test loss, test mse:', results)
        results_file = open('results_network.txt', 'a')
        results_file.write(loss_func + ' + ' + opt_func + ' + ' + str(EPOCHS) + '\ntest loss: ' + str(results[0]) + '   test mse: ' + str(results[1]) + '\n\n')
        results_file.close()

        predictions = model.predict(x=X)
        print(predictions[0])
        print(y_test[0])
        qr1 = constants.qr_config
        url = constants.input_url + binary_to_string(y_test[0])
        qr1.add_data(url)
        rounded_numbers = list(map(lambda x: round(x), predictions[0]))
        url = constants.input_url + binary_to_string(rounded_numbers)
        qr2 = constants.qr_config
        qr2.add_data(url)
        print(constants.matrix_acc(qr1.get_matrix(), qr2.get_matrix()))