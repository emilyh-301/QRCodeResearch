import qrcode
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from mappings import output_mapping
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

input_url = 'https://h3turing.vmhost.psu.edu?'
train_data_path = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data_path = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'


def load_my_data(path, num):
    file = open(path, 'r')
    data = np.loadtxt(file, delimiter=',', ndmin=2).reshape(num, 33, 33)
    file.close()
    return data

# my custom loss function
def QRCodeLoss(y_true, y_pred):
    '''
   @:param y_true: the correct query string to append
   @:param y_pred: list of numbers output by the neural network, list length = batch size
   '''

    # map the neural network prediction to a string
    map_pred = []
    for pred in y_pred:  # y_pred is a Tensor obj
        m = ''
        for x in pred:
            m += output_mapping[x]
        map_pred.append(m)

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=0,
    )

    new_y_pred = []
    for pred in map_pred:
        qr.add_data(input_url + pred)
        new_y_pred.append(qr.get_matrix())
        qr.clear()

    new_y_true = []
    for true in y_true:
        qr.add_data(input_url + true)
        new_y_true.append(qr.get_matrix())
        qr.clear()

    y_true = [[[float(value) for value in row] for row in matrix] for matrix in new_y_true]  # convert from booleans to floats for the loss function
    y_pred = [[[float(value) for value in row] for row in matrix] for matrix in new_y_pred]

    cc = CategoricalCrossentropy(from_logits=True)
    return cc(y_true, y_pred).numpy()


# efficient net https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0
model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(33, 33, 1),
    pooling=None,
    classes=30,
    classifier_activation=None,
)

# try to use the GPU
# tf.cuda()

model.compile(
    optimizer='adagrad',
    loss=QRCodeLoss,
    metrics='acc',
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
)
model.save('my_qr_model')

# training
X = load_my_data(train_data_path, 16000)  # numpy array of input QR codes
read_train_labels = open(train_labels, 'r')
Y = read_train_labels.read().split('\n')  # the corresponding appended query string
read_train_labels.close()
print('Training the model')
model.fit(x=X, y=np.asarray(Y), batch_size=128, epochs=100, validation_split=.2)

# testing
X = load_my_data(test_data_path, 4000)
read_test_labels = open(test_labels, 'r')
y_test = read_test_labels.read().split('\n')
read_test_labels.close()
print('Evaluate on test data')
results = model.evaluate(x=X, y=np.asarray(y_test), batch_size=128)
print('test loss, test acc:', results)


