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
EPOCHS = 20


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
    return tf.round(tf.sigmoid(x))


# efficient net https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0
model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(33, 33, 1),
    pooling=None,
    classes=180,
    classifier_activation=round_output,
)

model.compile(
    optimizer='adagrad',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics='acc',
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
)


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
history = model.fit(x=X, y=tf.convert_to_tensor(Y, dtype=tf.int32), epochs=EPOCHS, validation_split=.2)
model.save('my_qr_network')

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
predictions = model.predict(x=X)
# results = model.evaluate(x=X, y=tf.convert_to_tensor(y_test, dtype=tf.int32))
# print('test loss, test acc:', results)
# results_file = open('results_network.txt', 'a')
# results_file.write('test loss: ' + str(results[0]) + ' test acc: ' + str(results[1]))
# results_file.close()

print(predictions[0])