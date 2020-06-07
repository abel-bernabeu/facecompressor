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


def test_bit_streamer():
    from testfixtures import TempDirectory
    import os
    with TempDirectory() as d:
        d.create()
        filename = "tmp.txt"
        os.chdir(d.getpath("") )
        stream = BitStreamer()
        stream.put(1,4)
        stream.put(4,4)
        stream.put(66,8)
        stream.put(67,7)
        stream.write(filename)
        with open(filename, 'r') as file:
            data = file.read()
            assert(data == "ABC")
        stream.read(filename)
        value = stream.get(8)
        assert(value == 0x41)
        value = stream.get(4)
        assert(value == 0x2)
        value = stream.get(4)
        assert(value == 0x4)
        value = stream.get(8)
        assert(value == 0x43)
