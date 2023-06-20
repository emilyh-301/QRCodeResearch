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


def matrix_acc(qr1, qr2):
    """
    compares 2 qr code query strings
    :param qr1: first query string
    :param qr2: second query string
    :return: the percent of matching bits
    """
    count = 0
    for x in range(180):
        count += 1 if qr1[x] == qr2[x] else 0
    return count/180
