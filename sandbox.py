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
qr1.add_data('https://h3turing.vmhost.psu.edu?1234nnnnn')
qr1_matrix = [[float(value) for value in row] for row in qr1.get_matrix()]


qr2 = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)
qr2.add_data('https://h3turing.vmhost.psu.edu?123456789')
qr2_matrix = [[float(value) for value in row] for row in qr2.get_matrix()]


bce = BinaryCrossentropy(from_logits=False)
loss = bce(qr1_matrix, qr2_matrix).numpy()

print(loss)
