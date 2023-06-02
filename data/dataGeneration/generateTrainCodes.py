import numpy as np
import constants
from mappings import char_to_binary

# simplest QR Code with no border and box_size of 1
qr = constants.qr_config
# reverse the char to binary dict
binary_to_char = {value: key for key, value in char_to_binary.items()}


def binary_to_string(binary):
    """
    :param binary: a string of 0 and 1s
    :return: the corresponding alphanumeric string
    """
    string = ''
    for x in range(0, len(binary)-6, 6):
        b = binary[x:x+6]
        string += binary_to_char[b]
    return string

trainStrings = open('../train/queryStrings.txt', 'r')
trainCodes = open('../train/qrCodes.txt', 'w+')
while True:
    line = trainStrings.readline()
    if not line:
        break
    line = binary_to_string(line)
    qr.clear()
    qr.add_data(constants.input_url + line)
    matrix = qr.get_matrix()
    np.savetxt(trainCodes, np.asarray(matrix), fmt='%d', delimiter=',')
trainCodes.close()
trainStrings.close()
print('Done generating train codes')