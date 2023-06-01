import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
import os
import constants

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

train_data_path = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data_path = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'

BATCH_SIZE = 128
EPOCHS = 100


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


# keys = list(output_mapping.keys())
# values = [output_mapping[k] for k in keys]
# table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value='-1')


# efficient net https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0
model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(33, 33, 1),
    pooling=None,
    classes=180,
    classifier_activation=None,  # TODO: try softmax
)

model.compile(
    optimizer='adagrad',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics='acc',
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
)
# model.save('my_qr_network') TODO: fix this TypeError: LossFunctionWrapper.get_config() missing 1 required positional argument: 'self'

# training
X = load_my_data(train_data_path, constants.num_of_train_data)  # numpy array of input QR codes
read_train_labels = open(train_labels, 'r')
Y = read_train_labels.read().split('\n')  # the corresponding appended query string
read_train_labels.close()
print('Training the model')
history = model.fit(x=X, y=np.asarray(Y), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=.2)

# testing
X = load_my_data(test_data_path, constants.num_of_test_data)
read_test_labels = open(test_labels, 'r')
y_test = read_test_labels.read().split('\n')
read_test_labels.close()
print('Evaluate on test data')
results = model.evaluate(x=X, y=np.asarray(y_test), batch_size=BATCH_SIZE)
print('test loss, test acc:', results)
