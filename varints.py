import sys

# Adopted from
# https://github.com/bright-tools/varints

ONE_BYTE_LIMIT = 240
TWO_BYTE_LIMIT = 2287
THREE_BYTE_LIMIT = 67823

FOUR_BYTE_LIMIT = 16777215
FIVE_BYTE_LIMIT = 4294967295
SIX_BYTE_LIMIT = 1099511627775
SEVEN_BYTE_LIMIT = 281474976710655
EIGHT_BYTE_LIMIT = 72057594037927935
NINE_BYTE_LIMIT = 18446744073709551615
THREE_BYTE_HEADER = 249
FOUR_BYTE_HEADER = 250
FIVE_BYTE_HEADER = 251
SIX_BYTE_HEADER = 252
SEVEN_BYTE_HEADER = 253
EIGHT_BYTE_HEADER = 254
NINE_BYTE_HEADER = 255
BYTE_VALS = 256
SHORT_VALS = 65536

BUCKET_OFFSET = 2

minint = 0
maxint = NINE_BYTE_LIMIT

buckets = [{'limit': FOUR_BYTE_LIMIT,
            'header': FOUR_BYTE_HEADER},
           {'limit': FIVE_BYTE_LIMIT,
            'header': FIVE_BYTE_HEADER},
           {'limit': SIX_BYTE_LIMIT,
            'header': SIX_BYTE_HEADER},
           {'limit': SEVEN_BYTE_LIMIT,
            'header': SEVEN_BYTE_HEADER},
           {'limit': EIGHT_BYTE_LIMIT,
            'header': EIGHT_BYTE_HEADER},
           {'limit': NINE_BYTE_LIMIT,
            'header': NINE_BYTE_HEADER},
           ]


def writeToFile(payload, filename):
    with open(filename, "wb") as f:
        f.write(encode(payload))


def readFromFile(filename):
    with open(filename, "rb") as f:
        bytes = f.read()
    return decode(bytes)


def encode(num):
    return generic_encode(num, funcs)


def encode_int(num):
    if num < 0:
        raise ValueError("Negative numbers not handled")

    if num <= ONE_BYTE_LIMIT:
        ret_val = varint_storage(num)
    elif num <= TWO_BYTE_LIMIT:
        top = num - ONE_BYTE_LIMIT
        ret_val = varint_storage((top // BYTE_VALS) + ONE_BYTE_LIMIT + 1) + \
                  varint_storage(top % BYTE_VALS)
    elif num <= THREE_BYTE_LIMIT:
        top = num - (TWO_BYTE_LIMIT + 1)
        ret_val = varint_storage(THREE_BYTE_HEADER) + \
                  varint_storage(top // BYTE_VALS) + \
                  varint_storage(top % BYTE_VALS)
    else:
        start = 0

        # Work out how many bytes are needed to store this value
        while ((start < len(buckets)) and
               (num > buckets[start]['limit'])):
            start = start + 1

        if start == len(buckets):
            raise ValueError("Too large")

        ret_val = varint_storage(buckets[start]['header'])
        mod = (buckets[start]['limit'] + 1) // BYTE_VALS
        start = start + BUCKET_OFFSET

        while start >= 0:
            start = start - 1
            ret_val = ret_val + varint_storage(num // mod)
            num = num % mod
            mod = mod // BYTE_VALS

    return ret_val


def decode(num):
    res = generic_decode(num, funcs)
    if type(res) == int:  # Convert from scalar to 1-d vector
        res = [res]
    return res


def decode_val(num):
    bytes_used = 1
    first = store_to_num(num[0])
    if first <= ONE_BYTE_LIMIT:
        ret_val = first
    elif first < THREE_BYTE_HEADER:
        second = store_to_num(num[1])
        ret_val = ONE_BYTE_LIMIT + (BYTE_VALS * (first - (ONE_BYTE_LIMIT + 1))) + second
        bytes_used = 2
    elif first == THREE_BYTE_HEADER:
        second = store_to_num(num[1])
        third = store_to_num(num[2])
        ret_val = (TWO_BYTE_LIMIT + 1) + (BYTE_VALS * second) + third
        bytes_used = 3
    else:
        data_bytes = first - 247
        start = data_bytes - 1
        ret_val = 0
        i = 1

        mod = (buckets[start - BUCKET_OFFSET]['limit'] + 1) // BYTE_VALS

        while start >= 0:
            ret_val = ret_val + (mod * store_to_num(num[i]))
            i = i + 1
            start = start - 1
            mod = mod // BYTE_VALS

        bytes_used = data_bytes + 1

    return ret_val, bytes_used


funcs = {'decode_val': decode_val,
         'encode_int': encode_int}

if sys.version_info[0] > 2:
    def empty_varint_storage():
        return bytes()


    def varint_storage(b):
        return bytes((b,))


    def store_to_num(b):
        return b


    def num_types():
        return int
else:
    def empty_varint_storage():
        return ""


    def varint_storage(b):
        return chr(b)


    def store_to_num(b):
        return ord(b)


def dump(num):
    print("Len: {}".format(len(num)))
    for element in num:
        print("B: {}".format(store_to_num(element)))


def generic_encode(num, funcs):
    ret_val = None
    if isinstance(num, list):
        ret_val = encode_list(num, funcs)
    elif isinstance(num, num_types()):
        ret_val = funcs['encode_int'](num)
    return ret_val


def encode_list(num, funcs):
    ret_val = empty_varint_storage()
    for val in num:
        ret_val = ret_val + funcs['encode_int'](val)
    return ret_val


def generic_decode(num, funcs):
    ret_val = None
    if isinstance(num, (str, bytes)):
        ptr = 0
        while ptr < len(num):
            (int_val, bytes_used) = funcs['decode_val'](num[ptr:])
            ptr = ptr + bytes_used
            if ret_val is None:
                ret_val = int_val
            else:
                if isinstance(ret_val, num_types()):
                    ret_val = [ret_val]
                ret_val.append(int_val)
    return ret_val
