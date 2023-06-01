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

b = string_to_base64_binary('emily')

print(b)

s = binary_to_string(b)
print(s)

