import qrcode

num_of_train_data = 16000
num_of_test_data = 4000

input_url = 'https://h3turing.vmhost.psu.edu?'

qr_config = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)


def matrix_acc(qr1, qr2):
    """
    compares 2 qr code matrices
    :param qr1: first matrix
    :param qr2: second matrix
    :return: the percent of matching squares
    """
    total = 33 * 33
    count = 0
    for row in range(len(qr1)):
        for col in range(len(qr1[0])):
            count += 1 if qr1[row][col] == qr2[row][col] else 0
    return count / total
