import qrcode.image.svg
import qrcode
from PIL import Image

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=0,
)
qr.add_data('https://h3turing.vmhost.psu.edu?Abcdefghij1234567890abcdefghiJ')
qr.make(fit=True)

print(qr.get_matrix())


alphanumeric = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

img = qr.make_image(fill_color="black", back_color="white")
# img.show()


# print(pow(62,30)) 591222134364399413463902591994678504204696392694759424

# print(len(str(pow(2, 33 * 33))))   382 digits long


# img = qr.make_image(back_color=(255, 195, 235), fill_color=(55, 95, 35))

# method = 'basic'
# if method == 'basic':
#     # Simple factory, just a set of rects.
#     factory = qrcode.image.svg.SvgImage
# elif method == 'fragment':
#     # Fragment factory (also just a set of rects)
#     factory = qrcode.image.svg.SvgFragmentImage
# else:
#     # Combined path factory, fixes white space that may occur when zooming
#     factory = qrcode.image.svg.SvgPathImage
#
# img = qrcode.make('Some data here', image_factory=factory)