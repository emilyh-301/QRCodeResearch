import qrcode
import numpy as np

# simplest QR Code with no border and box_size of 1
qr = qrcode.QRCode(
    version=1,
    box_size=1,
    border=0,
)
qr.make(fit=True)

trainStrings = open('../test/queryStrings.txt', 'r')
trainCodes = open('../test/qrCodes.txt', 'a+')
while True:
    line = trainStrings.readline()
    if not line:
        break
    qr.clear()
    qr.add_data('https://h3turing.vmhost.psu.edu?' + line)
    matrix = qr.get_matrix()
    np.savetxt(trainCodes, np.asarray(matrix), fmt='%d')
trainCodes.close()
trainStrings.close()