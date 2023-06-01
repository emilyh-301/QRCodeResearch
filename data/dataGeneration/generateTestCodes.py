import numpy as np
import base64
import constants

# simplest QR Code with no border and box_size of 1
qr = constants.qr_config

def binary_to_string(binary):
    base64_bytes = int(binary, 2).to_bytes((len(binary) + 7) // 8, 'big')
    string = base64.b64decode(base64_bytes).decode('utf-8')
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