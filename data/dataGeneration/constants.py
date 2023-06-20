import qrcode

num_of_train_data = 80000
num_of_test_data = 20000

input_url = 'https://h3turing.vmhost.psu.edu?'

qr_config = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)