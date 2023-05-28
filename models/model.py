import qrcode
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from mappings import output_mapping
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
   @:param y_true: the input QR Code matrix
   @:param y_pred: index of Y as a Tensor, so we have to convert it
   '''
    print('shape of y_pred ' + y_pred.shape)
    print('type of y_pred ' + type(y_pred))
    # map the nn output to strings
    map_pred = ''
    for x in Y[int(y_pred)]:  # convert y_pred from a tensor to an int
        map_pred += output_mapping[x]
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=0,
    )
    qr.add_data(input_url + map_pred)
    # pred_matrix = qr.get_matrix()
    cc = CategoricalCrossentropy(from_logits=True)
    y_true = [[float(value) for value in row] for row in y_true]  # convert from booleans to floats for the loss function
    y_pred = [[float(value) for value in row] for row in qr.get_matrix()]
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
X = load_my_data(train_data_path, 16000)
read_train_labels = open(train_labels, 'r')
Y = read_train_labels.read().split('\n')
read_train_labels.close()
print('Training the model')
model.fit(x=X, y=np.asarray(list(range(0, 16000))), batch_size=128, epochs=100, validation_split=.2)

# testing
X = load_my_data(test_data_path, 4000)
read_test_labels = open(test_labels, 'r')
y_test = read_test_labels.read().split('\n')
read_test_labels.close()
print('Evaluate on test data')
results = model.evaluate(x=X, y=np.asarray(list(range(0, 4000))), batch_size=128)
print('test loss, test acc:', results)


