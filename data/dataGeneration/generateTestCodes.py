import qrcode
import numpy as np

# simplest QR Code with no border and box_size of 1
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)
#qr.make(fit=True)

testStrings = open('../test/queryStrings.txt', 'r')
testCodes = open('../test/qrCodes.txt', 'w+')
while True:
    line = testStrings.readline()
    if not line:
        break
    qr.clear()
    qr.add_data('https://h3turing.vmhost.psu.edu?' + line)
    matrix = qr.get_matrix()
    np.savetxt(testCodes, np.asarray(matrix), fmt='%d', delimiter=',')
testCodes.close()
testStrings.close()
print('Done generating test codes')