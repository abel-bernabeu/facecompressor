class BitStreamWriter:

    def __init__(self):
        self.bits = []

    def put(self, value, num_bits=1):
        mask = 1
        for index in range(num_bits):
            is_set = (value & mask) != 0
            self.bits.append(is_set)
            mask <<= 1

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
                    if (self.bits[byte_index * 8 + bit_index]):
                        value |= (1 << bit_index)
                data[byte_index] = value
            file.write(data)
            self.bits = []


def test_bit_stream_writer():
    from testfixtures import TempDirectory
    import os
    with TempDirectory() as d:
        d.create()
        filename = "tmp.txt"
        os.chdir(d.getpath("") )
        stream = BitStreamWriter()
        stream.put(1,4)
        stream.put(4,4)
        stream.put(66,8)
        stream.put(67,7)
        stream.write(filename)
        with open(filename, 'r') as file:
            data = file.read()
            assert(data == "ABC")