import secrets
from functools import partial
from mappings import char_to_binary
import constants

alphanumeric = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

# saves 30 char query strings in base 64 binary to train and test files


def string_to_base64_binary(input_string):
    binary = ''
    for x in input_string:
        binary += char_to_binary[x]
    return binary


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
