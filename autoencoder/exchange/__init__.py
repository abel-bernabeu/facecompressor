import math


exponent_bits = 11
mantissa_bits = 53


def to_sign_exponent_mantissa(value, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits):
    """
    Returns a triplet with a 1 bit sign, an 11 bits unsigned integer exponent and a 53 bits
    unsigned integer mantissa. The returned triplet can be used to fully reconstruct the
    fp64 value passed as an argument.
    """
    float_mantissa, float_exponent = math.frexp(value)
    if (float_mantissa >= 0):
        sign = 0
    else:
        sign = 1
    exponent = int(float_exponent + 2**(exponent_bits - 1))
    mantissa = int(abs(float_mantissa) * 2**mantissa_bits)
    return sign, exponent, mantissa


def from_sign_exponent_mantissa(sign, exponent, mantissa, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits):
    """
    Returns an fp64 from a 1 bit sign, an 11 bit unsigned integer exponent and a
    53 bit unsigned integer mantissa.
    """
    if (sign):
        signed_mantissa = - mantissa
    else:
        signed_mantissa = mantissa
    signed_exponent = exponent - 2**(exponent_bits - 1)
    norm_signed_mantissa = float(signed_mantissa) / float(2**mantissa_bits)
    return math.ldexp(norm_signed_mantissa, signed_exponent)


class BitStreamer:

    def __init__(self):
        self.bits = []
        self.getpos = 0

    def put(self, value, num_bits=1):
        mask = 1
        for index in range(num_bits):
            is_set = (value & mask) != 0
            self.bits.append(is_set)
            mask <<= 1

    def get(self, num_bits=1):
        mask = int(1)
        value = int(0)
        for index in range(num_bits):
            if self.bits[self.getpos]:
                value |= mask
            self.getpos += 1
            mask <<= 1
        if (self.getpos == len(self.bits)):
            self.bits = []
            self.getpos = 0
        return value

    def put_double(self, value):
        sign, exponent, mantissa = to_sign_exponent_mantissa(value)
        self.put(mantissa, mantissa_bits)
        self.put(exponent, exponent_bits)
        self.put(sign, 1)

    def get_double(self):
        mantissa = self.get(mantissa_bits)
        exponent = self.get(exponent_bits)
        sign = self.get(1)
        return from_sign_exponent_mantissa(sign, exponent, mantissa)

    def pad(self, num_bits=8):
        remainder_bits = len(self.bits) % num_bits
        if (remainder_bits > 0):
            padding_bits = num_bits - remainder_bits
            self.put(0, padding_bits)

    def write(self, filename):
        with open(filename, 'wb') as file:
            self.pad(8)
            num_bytes = len(self.bits) // int(8)
            data = bytearray(num_bytes)
            for byte_index in range(num_bytes):
                value = 0
                for bit_index in range(8):
                    if (self.get(1) != 0):
                        value |= (1 << bit_index)
                data[byte_index] = value
            file.write(data)
            self.bits = []

    def read(self, filename):
        with open(filename, 'r') as file:
            read_char = file.read(1)
            while read_char:
                self.put(ord(read_char), 8)
                read_char = file.read(1)


def save_tensors(filename, tensors, per_channel_min_value, per_channel_max_value, per_channel_numbits):
    stream = BitStreamer()
    stream.put(len(tensors), ord('T'))
    stream.put(len(tensors), ord('X'))
    stream.put(len(tensors), 8)
    for tensor in tensors:
        tensor.cpu()
        stream.put(tensor.shape[0], 16)
        stream.put(tensor.shape[1], 16)
        stream.put(tensor.shape[2], 16)
        stream.put(tensor.shape[3], 16)
        for batch_elem in range(tensor.shape[0]):
            for channel in range(tensor.shape[1]):
                numbits = per_channel_numbits[batch_elem][channel]
                stream.put_double(per_channel_min_value[batch_elem][channel])
                stream.put_double(per_channel_max_value[batch_elem][channel])
                stream.put(per_channel_numbits[batch_elem][channel], 6)
                for row in range(tensor.shape[2]):
                    for col in range(tensor.shape[3]):
                        value = int(tensor[batch_elem][channel][row][col])
                        stream.put(value, numbits)
    stream.write(filename)


def load_tensors(filename, tensors, per_channel_min_value, per_channel_max_value, per_channel_numbits):
    result = []
    stream = BitStreamer()
    stream.put(len(tensors), ord('T'))
    stream.put(len(tensors), ord('X'))
    stream.put(len(tensors), 8)
    for tensor in tensors:
        tensor.cpu()
        stream.put(tensor.shape[0], 16)
        stream.put(tensor.shape[1], 16)
        stream.put(tensor.shape[2], 16)
        stream.put(tensor.shape[3], 16)
        for batch_elem in range(tensor.shape[0]):
            for channel in range(tensor.shape[1]):
                numbits = per_channel_numbits[batch_elem][channel]
                stream.put_double(per_channel_min_value[batch_elem][channel])
                stream.put_double(per_channel_max_value[batch_elem][channel])
                stream.put(per_channel_numbits[batch_elem][channel], 6)
                for row in range(tensor.shape[2]):
                    for col in range(tensor.shape[3]):
                        value = int(tensor[batch_elem][channel][row][col])
                        stream.put(value, numbits)
    stream.write(filename)


def test_float_storage():
    """Check that the storage of a float as sign, exponent and mantissa
     and way back produces exactly the original number.    
    """
    values = [2.3434, 124012.2323209999, -12.39212445433389]

    for value in values:
      sign, exp, mantissa = to_sign_exponent_mantissa(value)
      restored_value = from_sign_exponent_mantissa(sign, exp, mantissa)
      print(restored_value)
      assert(value == restored_value)


def test_bit_streamer():
    from testfixtures import TempDirectory
    import os
    with TempDirectory() as d:

        # Create temporary directory
        d.create()
        filename = "tmp.txt"
        os.chdir(d.getpath("") )

        stream = BitStreamer()

        # Write some content
        stream.put(1,4)
        stream.put(4,4)
        stream.put(66,8)
        stream.put(67,7)
        stream.write(filename)

        # Read the contents as a file
        with open(filename, 'r') as file:
            data = file.read()
            assert(data == "ABC")

        # Read the contents as a stream
        stream.read(filename)
        value = stream.get(8)
        assert(value == 0x41)
        value = stream.get(4)
        assert(value == 0x2)
        value = stream.get(4)
        assert(value == 0x4)
        value = stream.get(8)
        assert(value == 0x43)

        # Check that floats can be put and get without loss
        values = [2.33434, -1012.2323209999, -12.39212445433389]
        for value in values:
            stream.put_double(value)
            read_value = stream.get_double()
            assert(value == read_value)