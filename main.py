import qrcode.image.svg
import qrcode
import numpy as np
from PIL import Image

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)
qr.add_data('https://h3turing.vmhost.psu.edu?Abcdefghij1234567890abcdefghiJ')
#qr.make(fit=True)
qr.clear()
print(qr.border, qr.box_size)
qr.add_data('https://h3turing.vmhost.psu.edu?' + 'Abcdefghij1234567890abcdefghiJ2222')
print(qr.data_list)

print(qr.get_matrix())
file = open('try.txt', 'w+')
np.savetxt(file, np.asarray(qr.get_matrix()), fmt='%d', delimiter='*')
file.close()
read_file = open('try.txt', 'r')
print(np.loadtxt(read_file, delimiter='*'))
read_file.close()
print('*******************************')
file = open('try.txt', 'a+')
#:qfile.write('*')
np.savetxt(file, np.asarray(qr.get_matrix()), fmt='%d', delimiter='*')
file.close()
read_file = open('try.txt', 'r')
info = np.loadtxt(read_file, delimiter='*')
print(info[1])
print('shape' + str(info.shape))
read_file.close()

alphanumeric = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

img = qr.make_image(fill_color="black", back_color="white")
# img.show()


# print(pow(62,30)) 591222134364399413463902591994678504204696392694759424

# print(len(str(pow(2, 33 * 33))))   382 digits long


# img = qr.make_image(back_color=(255, 195, 235), fill_color=(55, 95, 35))

