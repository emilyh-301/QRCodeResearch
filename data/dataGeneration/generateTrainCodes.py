import constants
import numpy as np
import base64

# simplest QR Code with no border and box_size of 1
qr = constants.qr_config

def binary_to_string(binary):
    base64_bytes = int(binary, 2).to_bytes((len(binary) + 7) // 8, 'big')
    string = base64.b64decode(base64_bytes).decode('utf-8')
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