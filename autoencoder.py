import torch

def train():
    if (not torch.cuda.is_available()):
        print('Error: CUDA is needed for training')
        sys.exit(0)
    print ('TODO: training')

def encode():
    print ('TODO: encoding')

def decode():
    print('TODO: decoding')
