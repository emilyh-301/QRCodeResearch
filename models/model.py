import qrcode
import tensorflow as tf
from tensorflow.keras.losses import Loss
from mappings import output_mapping
import numpy as np

input_url = 'https://h3turing.vmhost.psu.edu?'
train_data_path = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data_path = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'
# load the labels
read_train_labels = open(train_labels, 'r')
y_train = read_train_labels.read().split('\n')
read_train_labels.close()
read_test_labels = open(test_labels, 'r')
y_test = read_test_labels.read().split('\n')
read_test_labels.close()


def load_my_data(path, num):
    file = open(path, 'r')
    data = np.loadtxt(file, delimiter=',', ndmin=2).reshape(num, 33, 33)
    file.close()
    return data


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

model.compile(
    optimizer='adagrad',
    loss=None,
    metrics='acc',
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
)
model.save('my_qr_model')

# training
X = load_my_data(train_data_path, 16000)
print('Training the model')
model.fit(x=X, y=y_train, batch_size=128, epochs=100)

# testing
X = load_my_data(test_data_path, 4000)
print('Evaluate on test data')
results = model.evaluate(x=X, y=y_test, batch_size=128)
print('test loss, test acc:', results)


# my own loss function
class QRCodeLoss(Loss):
    '''
    @:param y_true: the input QR Code
    @:param y_pred: the output 30 character sequence
    '''

    def call(self, y_true, y_pred):
        # print('shape of y_pred ' + y_pred.shape)
        # print('type of y_pred ' + type(y_pred))
        # map the nn output to strings
        map_pred = ''
        for x in y_pred:
            map_pred += output_mapping[x]
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=0,
        )
        qr.add_data(input_url + map_pred)
        # pred_matrix = qr.get_matrix()
        return Loss.BinaryCrossentropy(y_true, qr.get_matrix())
