import torch


class Quantize(torch.nn.Module):

    def __init__(self):
        super(Quantize, self).__init__()

    def forward(self, x, quantization_select = None, per_channel_num_bits = None):
        """
        During inference quantizes the i-th channel from x according to the
        number of bits requested in per_channel_num_bits[i]

        During training for each value from x it randomly picks between doing
        the linear quantization requested for its channel, as for the case of
        inference, or it applies the identity. The random choices, one per
        element in x, are provided through the quantization_select input tensor.

        If no per_channel_num_bits parameter is passed, then 8 bits are assumed
        for all the channels.

        The quantization_select tensor is to be set up with random binary values
        from the CPU side. More precisely the random values are expected to be
        setup by the data loader when providing a new training sample. The
        random_choices tensor is ignored for inference.
        """

        batch_dim_index = 0
        channels_dim_index = 1
        rows_dim_index = 2
        cols_dim_index = 3

        batch = x.size()[batch_dim_index]
        channels  = x.size()[channels_dim_index]
        height = x.size()[rows_dim_index]
        width  = x.size()[cols_dim_index]

        if quantization_select is None:
            quantization_select = torch.ones(batch, channels)

        if per_channel_num_bits is None:
            per_channel_num_bits = 8.0 * torch.ones(batch, channels)

        per_row_min, _ = torch.min(x, cols_dim_index)
        per_channel_min, _ = torch.min(per_row_min, rows_dim_index)

        per_row_max, _ = torch.max(x, cols_dim_index)
        per_channel_max, _ = torch.max(per_row_max, rows_dim_index)

        per_channel_range = per_channel_max - per_channel_min

        per_column_replicable_scale = torch.reciprocal(per_channel_range.reshape(batch, channels, 1, 1))

        per_column_replicable_min = per_channel_min.reshape(batch, channels, 1, 1)

        normalized_x = (x - per_column_replicable_min) * per_column_replicable_scale

        bases = 2.0 * torch.ones(batch, channels, height, width)
        exponents = per_channel_num_bits.reshape(batch, channels, 1, 1)
        ones = torch.ones(batch, channels, height, width)

        per_column_quantized_range = torch.pow(bases, exponents) - ones
        in_quantization_range_x = normalized_x * per_column_quantized_range
        quantized_x = torch.round(in_quantization_range_x)

        if self.training:
          result = quantization_select * quantized_x + (ones - quantization_select) * in_quantization_range_x
        else:
          result = quantized_x

        return result, per_channel_min, per_channel_max, per_channel_num_bits


class Dequantize(torch.nn.Module):

    def __init__(self):
        super(Dequantize, self).__init__()

    def forward(self, quantized_x, per_channel_min, per_channel_max, per_channel_num_bits):
        """
        Remaps a quantized tensor to its original ranges. The input parameters
        are exactly the same return values produced by the Quantize module, so
        the output from Quantized can be plugged as input for Dequantize.
        """

        batch_dim_index = 0
        channels_dim_index = 1
        rows_dim_index = 2
        cols_dim_index = 3

        batch = quantized_x.size()[batch_dim_index]
        channels  = quantized_x.size()[channels_dim_index]
        height = quantized_x.size()[rows_dim_index]
        width  = quantized_x.size()[cols_dim_index]

        bases = 2.0 * torch.ones(batch, channels, height, width)
        exponents = per_channel_num_bits.reshape(batch, channels, 1, 1)
        ones = torch.ones(batch, channels, height, width)
        quantized_range = torch.pow(bases, exponents) - ones

        normalized_x = quantized_x * torch.reciprocal(quantized_range)

        per_channel_scale = per_channel_max - per_channel_min
        per_column_replicable_scale = per_channel_scale.reshape(batch, channels, 1, 1)
        per_column_replicable_min = per_channel_min.reshape(batch, channels, 1, 1)

        result = normalized_x * per_column_replicable_scale + per_column_replicable_min

        return result


def test_quantize_dequantize_module():

    quant_module = Quantize()

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

    quantization_select = torch.tensor(
        [
            # Batch element 0
            [
                [  # Channel 0 of batch element 0
                    [1., 0, 1, 0],
                    [0, 1, 0, 1],
                ],
                [  # Channel 1 of batch element 0
                    [1., 0, 1, 0],
                    [0, 1, 0, 1],
                ],
                [  # Channel 3 of batch element 0
                    [1., 0, 1, 0],
                    [0, 1, 0, 1],
                ],
            ],
            # Batch element 1
            [
                [  # Channel 0 of batch element 1
                    [1., 0, 1, 0],
                    [0, 1, 0, 1],
                ],
                [  # Channel 1 of batch element 1
                    [1., 0, 1, 0],
                    [0, 1, 0, 1],
                ],
                [  # Channel 3 of batch element 1
                    [1., 0, 1, 0],
                    [0, 1, 0, 1],
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

    result, per_channel_min, per_channel_max, result_per_channel_num_bits = quant_module(x, quantization_select,
                                                                                         per_channel_num_bits)

    expected_result = torch.tensor(
        [[[[0.0000e+00, 1.4286e-01, 0.0000e+00, 4.2857e-01],
           [5.7143e-01, 1.0000e+00, 8.5714e-01, 1.0000e+00]],

          [[0.0000e+00, 4.2857e-01, 1.0000e+00, 1.2857e+00],
           [1.7143e+00, 2.0000e+00, 2.5714e+00, 3.0000e+00]],

          [[0.0000e+00, 2.1429e+00, 4.0000e+00, 6.4286e+00],
           [8.5714e+00, 1.1000e+01, 1.2857e+01, 1.5000e+01]]],

         [[[0.0000e+00, 3.6429e+01, 7.3000e+01, 1.0929e+02],
           [1.4571e+02, 1.8200e+02, 2.1857e+02, 2.5500e+02]],

          [[0.0000e+00, 9.3621e+03, 1.8724e+04, 2.8086e+04],
           [3.7449e+04, 4.6811e+04, 5.6173e+04, 6.5535e+04]],

          [[0.0000e+00, 1.4286e-01, 0.0000e+00, 4.2857e-01],
           [5.7143e-01, 1.0000e+00, 8.5714e-01, 1.0000e+00]]]])

    expected_per_channel_min = torch.tensor(
        [[1., 9., 17.],
         [25., 41., 57.]])

    expected_per_channel_max = torch.tensor(
        [[8., 16., 24.],
         [39., 55., 71.]])

    assert torch.allclose(result, expected_result, rtol=0.001)
    assert torch.allclose(per_channel_min, expected_per_channel_min)
    assert torch.allclose(per_channel_max, expected_per_channel_max)
    assert torch.allclose(result_per_channel_num_bits, per_channel_num_bits)

    dequantize_module = Dequantize()
    result2 = dequantize_module(result, per_channel_min, per_channel_max, result_per_channel_num_bits)

    expected_result2 = torch.tensor(
        [[[[1.0000, 2.0000, 1.0000, 4.0000],
           [5.0000, 8.0000, 7.0000, 8.0000]],

          [[9.0000, 10.0000, 11.3333, 12.0000],
           [13.0000, 13.6667, 15.0000, 16.0000]],

          [[17.0000, 18.0000, 18.8667, 20.0000],
           [21.0000, 22.1333, 23.0000, 24.0000]]],

         [[[25.0000, 27.0000, 29.0078, 31.0000],
           [33.0000, 34.9922, 37.0000, 39.0000]],

          [[41.0000, 43.0000, 44.9999, 47.0000],
           [49.0000, 51.0001, 53.0000, 55.0000]],

          [[57.0000, 59.0000, 57.0000, 63.0000],
           [65.0000, 71.0000, 69.0000, 71.0000]]]])

    assert torch.allclose(result2, expected_result2)
