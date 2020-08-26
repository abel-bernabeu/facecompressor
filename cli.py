import argparse
import autoencoder


def addTrainablesArg(parser):
    parser.add_argument('--model', dest='model', help='Trained model')

def addExchangeArg(parser):
    parser.add_argument('--exchange', dest='exchange', help='File with exchanged data', required=True)


parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest="action")

encode_parser = subparsers.add_parser('encode')
addTrainablesArg(encode_parser)
encode_parser.add_argument('--input', dest='input', help='Input image file name', required=True)
addExchangeArg(encode_parser)

decode_parser = subparsers.add_parser('decode')
addTrainablesArg(decode_parser)
addExchangeArg(decode_parser)
decode_parser.add_argument('--output', dest='output', help='Output image file name', required=True)

opts = parser.parse_args()

if opts.action == 'encode':
    autoencoder.encode(opts.model, opts.input, opts.exchange)
elif opts.action == 'decode':
    autoencoder.decode(opts.model, opts.exchange, opts.output)