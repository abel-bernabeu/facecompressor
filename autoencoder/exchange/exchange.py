import math
import torch
import autoencoder.models.quantization


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
        with open(filename, 'rb') as file:
            read_char = file.read(1)
            while read_char:
                self.put(ord(read_char), 8)
                read_char = file.read(1)


def wrap_in_a_single_element_batch(tensor):
    sizes = [1]
    for element in tensor.shape:
        sizes.append(element)
    return tensor.reshape(*sizes)


revision = 0
tensor_start_marker  = 12349
tensor_end_marker = 29913


def save(filename, model_id, tensor_list, per_channel_num_bits_list, batch_element):

    quantize = autoencoder.models.quantization.Quantize()

    stream = BitStreamer()

    stream.put(ord('Q'), 8)
    stream.put(ord('T'), 8)
    stream.put(ord('X'), 8)

    stream.put(revision, 16)

    stream.put(model_id, 16)

    stream.put(len(tensor_list), 16)

    # Stream the tensors
    for tensor_index, tensor in enumerate(tensor_list):

        # Write a marker for aiding on debugging issues of save/load desynchronization
        stream.put(tensor_start_marker, 16)

        # Quantize
        tensor = wrap_in_a_single_element_batch(tensor[batch_element])
        tensor = tensor.to('cpu')
        quantization_select = torch.ones(tensor.shape)
        per_channel_num_bits = wrap_in_a_single_element_batch(per_channel_num_bits_list[tensor_index][batch_element])

        quant_tensor, per_channel_min, per_channel_max, _ = \
            quantize(tensor, quantization_select, per_channel_num_bits)

        # Write number of channels
        stream.put(quant_tensor.shape[1], 16)

        # Write height
        stream.put(quant_tensor.shape[2], 16)

        # Write width
        stream.put(quant_tensor.shape[3], 16)

        # per_channel_num_bits shape and quant_tensor should have the same batch_size and num_channels
        assert (per_channel_num_bits.shape[0] == quant_tensor.shape[0])
        assert (per_channel_num_bits.shape[1] == quant_tensor.shape[1])

        # The expected per_channel_num_bits shape is batch_size x num_channels
        assert (per_channel_num_bits.shape[0] == 1)
        assert (len(per_channel_num_bits.shape) == 2)

        # Stream the channels
        for channel in range(quant_tensor.shape[1]):

            # Write the min and max values
            stream.put_double(per_channel_min[0][channel])
            stream.put_double(per_channel_max[0][channel])

            # Write the number of bits per component
            num_bits = int(per_channel_num_bits[0][channel])
            stream.put(num_bits, 6)

            # Write all the components
            for row in range(quant_tensor.shape[2]):
                for col in range(quant_tensor.shape[3]):
                    value = int(quant_tensor[0][channel][row][col])
                    stream.put(value, num_bits)

        # Write a marker for aiding on debugging issues of save/load desynchronization
        stream.put(tensor_end_marker, 16)

    stream.write(filename)


def load(filename):

    dequantize = autoencoder.models.quantization.Dequantize()

    stream = BitStreamer()
    stream.read(filename)

    # Read the header
    header1 = stream.get(8)
    header2 = stream.get(8)
    header3 = stream.get(8)

    is_valid_header = header1 == ord('Q') and header2 == ord('T') and header3 == ord('X')
    if not is_valid_header:
        raise Exception("Wrong header")

    read_format_revision = stream.get(16)

    if not read_format_revision is revision:
        raise Exception("Wrong format revision")

    # Read the model id
    model_id = stream.get(16)

    # Read the number of tensors
    num_tensors = stream.get(16)

    tensor_list = []

    for tensor_index in range(num_tensors):

        # Read a marker for aiding on debugging issues of save/load desynchronization
        marker = stream.get(16)
        assert(marker == tensor_start_marker)

        # Read number of channels
        num_channels = stream.get(16)

        # Read height
        height = stream.get(16)

        # Read width
        width = stream.get(16)

        # Create a tensor for number of bits per channel
        per_channel_num_bits = torch.zeros(1, num_channels)

        # Create a tensor for number of bits per channel
        per_channel_min = torch.zeros(1, num_channels)

        # Create a tensor for number of bits per channel
        per_channel_max = torch.zeros(1, num_channels)

        # Create a new tensor with zeros
        quant_tensor = torch.zeros(1, num_channels, height, width)

        # Read the channels
        for channel in range(num_channels):

            # Read the min and max values
            min = stream.get_double()
            max = stream.get_double()

            # Read the number of bits per component
            num_bits = stream.get(6)

            # Store the min
            per_channel_min[0][channel] = min

            # Store the max
            per_channel_max[0][channel] = max

            # Store the number of bits
            per_channel_num_bits[0][channel] = num_bits

            # Store all the components
            for row in range(height):
                for col in range(width):
                    value = stream.get(num_bits)
                    quant_tensor[0][channel][row][col] = float(value)

        # Read a marker for aiding on debugging issues of save/load desynchronization
        marker = stream.get(16)
        assert(marker == tensor_end_marker)

        # Dequantize
        tensor = dequantize(quant_tensor, per_channel_min, per_channel_max, per_channel_num_bits)

        # Append the dequantized tensor
        tensor_list.append(tensor)

    return model_id, tensor_list


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

        # Check that floats can be put and got without loss
        values = [2.33434, -1012.2323209999, -12.39212445433389]
        for value in values:
            stream.put_double(value)
            read_value = stream.get_double()
            assert(value == read_value)


def save_load_and_compare_test_case(filename, model_id, tensor_list, per_channel_num_bits_list, batch_element):

    save(filename, model_id, tensor_list, per_channel_num_bits_list, batch_element)
    loaded_model_id, loaded_tensor_list = load(filename)

    assert(model_id == loaded_model_id)
    assert(len(tensor_list) == len(loaded_tensor_list))

    # Create a quantizer and a dequantizer
    quantize = autoencoder.models.quantization.Quantize()
    dequantize = autoencoder.models.quantization.Dequantize()

    for tensor_index, tensor in enumerate(tensor_list):

        # Quantize and dequantize, then compare with the loaded tensor
        tensor = wrap_in_a_single_element_batch(tensor[batch_element])
        quantization_select = torch.ones(tensor.shape)
        per_channel_num_bits = wrap_in_a_single_element_batch(per_channel_num_bits_list[tensor_index][batch_element])

        quant_tensor, per_channel_min, per_channel_max, result_per_channel_num_bits = \
            quantize(tensor, quantization_select, per_channel_num_bits)

        reconstructed = dequantize(quant_tensor, per_channel_min, per_channel_max, result_per_channel_num_bits)

        loaded_tensor = loaded_tensor_list[0][tensor_index]

        assert(torch.allclose(reconstructed, loaded_tensor))


def test_save_load():

    from testfixtures import TempDirectory
    import os
    with TempDirectory() as d:

        # Create temporary directory
        d.create()
        filename = "exchange.qtx"
        os.chdir(d.getpath("") )

        # A input tensor for testing with batch=2, channels=3, height=2, width=4

        x = torch.tensor(
            [
                # Batch element 0
                [
                    [  # Channel 0 of batch element 0
                        [1., 2, 3, 4],
                        [5, 6, 7, 8],
                    ],
                    [  # Channel 1 of batch element 0
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ],
                    [  # Channel 3 of batch element 0
                        [17, 18, 19, 20],
                        [21, 22, 23, 24],
                    ],
                ],
                # Batch element 1
                [
                    [  # Channel 0 of batch element 1
                        [25, 27, 29, 31],
                        [33, 35, 37, 39],
                    ],
                    [  # Channel 1 of batch element 1
                        [41, 43, 45, 47],
                        [49, 51, 53, 55],
                    ],
                    [  # Channel 3 of batch element 1
                        [57, 59, 61, 63],
                        [65, 67, 69, 71],
                    ],
                ],
            ],
        )

        per_channel_num_bits = torch.tensor(
            [
                # Batch element 0
                [
                    # Channel 0 of batch element 0
                    1.,
                    # Channel 1 of batch element 0
                    2,
                    # Channel 3 of batch element 0
                    4,
                ],
                # Batch element 1
                [
                    # Channel 0 of batch element 1
                    8,
                    # Channel 1 of batch element 1
                    16,
                    # Channel 1 of batch element 1
                    1,
                ],
            ],
        )

        model_id = 6789
        tensor_list = [x]
        per_channel_num_bits_list = [per_channel_num_bits]
        batch_element = 1

        save_load_and_compare_test_case(filename, model_id, tensor_list, per_channel_num_bits_list, batch_element)
