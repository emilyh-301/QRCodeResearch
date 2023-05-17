import qrcode
import secrets
from functools import partial

alphanumeric = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

# 20,000 total
# 16,000 train
#  4,000 test
def produce_amount_keys(amount_of_keys, length=30):
    keys = set()
    pickchar = partial(secrets.choice, alphanumeric)
    while len(keys) < amount_of_keys:
        keys |= {''.join([pickchar() for _ in range(length)]) for _ in range(amount_of_keys - len(keys))}
    train = open('../train/queryStrings.txt', 'a')
    test = open('../test/queryStrings.txt', 'a')
    count = 0
    for key in keys:
        if count % 5 == 0:
            test.write(key + '\n')
        else:
            train.write((key + '\n'))
        count += 1
    train.close()
    test.close()
    print('query strings done')

produce_amount_keys(20000, 30)

# simplest QR Code with no border and box_size of 1
qr = qrcode.QRCode(
    version=1,
    box_size=1,
    border=0,
)
qr.make(fit=True)
