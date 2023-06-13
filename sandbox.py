import math

import qrcode
import base64

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

def string_to_base64_binary(input_string):
    string_bytes = input_string.encode('utf-8')
    base64_bytes = base64.b64encode(string_bytes)
    binary = bin(int.from_bytes(base64_bytes, 'big'))
    return binary

def binary_to_string(binary):
    base64_bytes = int(binary, 2).to_bytes((len(binary) + 7) // 8, 'big')
    string = base64.b64decode(base64_bytes).decode('utf-8')
    return string

b = string_to_base64_binary('emilyemilyemily')
print(len(b))
print(b)

s = binary_to_string(b)
print(s)


Y = ['1234567890', '1234567980']
newY = []
for y in Y:
    newY.append([int(char) for char in y])
print(newY)

c = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]
bo = '1010100010001110101001001010000010101010011000001110100010101000101101001101100010100100110100101010001001100010100001001011001011000100011001101011010011101000101000101010111011101110011001001100001010010000100111000110101010101100101010101101000001100110101011001000110011011000011000101010110010001010011100100110001'
print(len(bo))


from keras import backend as K
import numpy as np
def round_output(x):
    return np.floor(K.sigmoid(x) + .5)

o = [[.4,.7,.2], [0,.9,.1]]

print(round_output(o))

import tensorflow as tf
m = tf.keras.metrics.MeanSquaredError()
m.update_state([[1,0,1,0]], [[.9,.9,.9,.9]])
print(m.result().numpy())
