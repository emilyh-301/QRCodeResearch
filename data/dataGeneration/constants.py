import qrcode

num_of_train_data = 8 # 16000
num_of_test_data = 2 # 4000

input_url = 'https://h3turing.vmhost.psu.edu?'

qr_config = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)