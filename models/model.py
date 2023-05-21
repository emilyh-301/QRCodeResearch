import qrcode
import tensorflow as tf
from tensorflow.keras.losses import Loss
from mappings import output_mapping

input_url = 'https://h3turing.vmhost.psu.edu?'
train_data = '../data/train/qrCodes.txt'
train_labels = '../data/train/queryStrings.txt'
test_data = '../data/test/qrCodes.txt'
test_labels = '../data/test/queryStrings.txt'
x_train = None
x_test = None
y_train = open(train_labels, 'r').read().split('\n')
y_test = open(test_labels, 'r').read().split('\n')

# efficient net https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0
model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(1, 33, 33),
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

print('Training the model')
model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, validation_split=.2)

print('Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

class QRCodeLoss(Loss):
    '''
    @:param y_true: the input QR Code
    @:param y_pred: the output 30 character sequence
    '''
    def call(self, y_true, y_pred):
        print('shape of y_pred ' + y_pred.shape)
        print('type of y_pred ' + type(y_pred))
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
        #pred_matrix = qr.get_matrix()
        return Loss.BinaryCrossentropy(y_true, qr.get_matrix())
