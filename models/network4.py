import tensorflow as tf
import numpy as np
import os
import constants
from tensorflow.keras import losses, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from plot_graph import plot_performance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

train_data_path = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data_path = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'

BATCH_SIZE = 64
EPOCHS = 100


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


def _create_model(opt, ha='relu', oa='sigmoid', l='mean_squared_error') -> models.Sequential:
    hidden_activation = ha
    output_activation = oa
    optimizer = opt
    model = models.Sequential()

    model.add(Flatten(input_shape=(33, 33, 1)))

    # dense layers
    model.add(Dense(1089, activation=hidden_activation))
    model.add(Dense(2000, activation=hidden_activation))
    model.add(Dense(3000, activation=hidden_activation))
    model.add(Dense(3800, activation=hidden_activation))
    model.add(Dense(3000, activation=hidden_activation))
    model.add(Dense(2000, activation=hidden_activation))
    model.add(Dense(1000, activation=hidden_activation))

    # output layer
    model.add(Dense(180, activation=output_activation))

    # compile
    model.compile(
        optimizer=optimizer,
        loss=l,
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model


# model = _create_model(opt='adagrad')

# loss_funcs = ['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error', 'binary_crossentropy']
# opt_funcs = ['adagrad', 'adamax', 'adam', 'sgd', 'adadelta']
opt_funcs = ['adam']
loss_funcs = ['mean_squared_error']
for loss_func in loss_funcs:
    for opt_func in opt_funcs:
        model = _create_model(opt=opt_func, l=loss_func)
        # TODO load weights
        print(model.summary())

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
        model.save_weights('my_qr_network4_flat')
        plot_performance(history, title='plot_4_flatten' + loss_func + '_' + opt_func)

        # testing
        print('************************ Evaluate 4 on test data')
        X, y_test = load_test_data()
        results = model.evaluate(x=X, y=tf.convert_to_tensor(y_test, dtype=tf.int32))

        # write results to console and file
        print('test loss, test mse:', results)
        results_file = open('results_network4.txt', 'a')
        results_file.write('FLAT ' + loss_func + ' + ' + opt_func + ' + ' + str(EPOCHS) + '\ntest loss: ' + str(
            results[0]) + '   test mse: ' + str(results[1]) + '\n\n')
        results_file.close()

        predictions = model.predict(x=X)
        print(predictions[0])
        print(y_test[0])

        total_acc = 0
        for x in range(len(predictions)):
            rounded_numbers = list(map(lambda x: round(x), predictions[x]))
            total_acc += constants.matrix_acc(y_test[x], rounded_numbers)
        print('Overall accuracy of the 180 bits for test data: ')
        print(total_acc / constants.num_of_test_data)

        results_file = open('results_network4.txt', 'a+')
        results_file.write('FLATTEN test accuracy: ' + str(total_acc / constants.num_of_test_data) + '\n\n')
        results_file.close()

        print(model.summary())
