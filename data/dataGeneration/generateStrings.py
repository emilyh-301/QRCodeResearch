import secrets
from functools import partial
import base64
import constants

alphanumeric = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

# saves 30 char query strings in base 64 binary to train and test files

# 20,000 total
# 16,000 train
#  4,000 test
def produce_amount_keys(amount_of_keys, length=30):
    keys = set()
    pickchar = partial(secrets.choice, alphanumeric)
    while len(keys) < amount_of_keys:
        keys |= {''.join([pickchar() for _ in range(length)]) for _ in range(amount_of_keys - len(keys))}
    train = open('../train/queryStrings.txt', 'w+')
    test = open('../test/queryStrings.txt', 'w+')
    count = 0
    for key in keys:
        binary = string_to_base64_binary(key)
        # 20% of the data will be testing data, the other 80% will be training data
        if count % 5 == 0:
            test.write(binary + '\n')
        else:
            train.write((binary + '\n'))
        count += 1
    train.close()
    test.close()
    print('query strings done')

produce_amount_keys(constants.num_of_train_data + constants.num_of_test_data, 30)

def string_to_base64_binary(input_string):
    string_bytes = input_string.encode('utf-8')
    base64_bytes = base64.b64encode(string_bytes)
    binary = bin(int.from_bytes(base64_bytes, 'big'))
    return binary
