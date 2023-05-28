import qrcode
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

# tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )

# print(tf.config.list_physical_devices('GPU'))
# simplest QR Code with no border and box_size of 1
qr1 = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)
qr1.add_data('https://h3turing.vmhost.psu.edu?1234')

print('type of qr matrix')
print(type(qr1.get_matrix()[1][1]))

qr1_matrix = [[int(value) for value in row] for row in qr1.get_matrix()]

print(qr1_matrix)



bce = BinaryCrossentropy()
loss = bce(qr1_matrix, qr1_matrix)

print(loss)
