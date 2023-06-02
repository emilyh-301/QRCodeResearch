import numpy as np
import constants
from mappings import char_to_binary

# simplest QR Code with no border and box_size of 1
qr = constants.qr_config
# reverse the char to binary dict
binary_to_char = {value: key for key, value in char_to_binary.items()}


def binary_to_string(binary):
    """
    :param binary: a string in the form 0b(0,1)*
    :return: the corresponding alphanumeric string
    """
    string = ''
    binary = binary[2:]
    for x in range(0, len(binary)-6, 6):
        b = binary[x:x+6]
        string += binary_to_char[b]
    return string


testStrings = open('../test/queryStrings.txt', 'r')
testCodes = open('../test/qrCodes.txt', 'w+')
while True:
    line = testStrings.readline()
    if not line:
        break
    line = binary_to_string(line)
    qr.clear()
    qr.add_data(constants.input_url + line)
    matrix = qr.get_matrix()
    np.savetxt(testCodes, np.asarray(matrix), fmt='%d', delimiter=',')
testCodes.close()
testStrings.close()
print('Done generating test codes')