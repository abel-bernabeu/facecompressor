import torch
import torchvision.transforms as transforms
from PIL import Image
import autoencoder.models
import autoencoder.exchange


class Context:

    def __init__(self, model):
        # Instantiate the model
        self.model = autoencoder.models.Compressor()

        # Load model from the checkpoint
        checkpoint = torch.load(model)
        self.model.load_state_dict(checkpoint['best_model'])

    def encode(self, input, exchange):

        # Load image and convert to tensor
        image = Image.open(input)
        to_tensor = transforms.Compose([transforms.ToTensor()])
        x = to_tensor(image)

        # Pack the tensor in a single-element batch
        num_channels = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]
        single_element_batch = x.clone().detach().reshape(1, num_channels, height, width)

        # Encode
        output = self.model.encoder(single_element_batch)

        # Save the intermediate tensor
        tensor_list = [output]
        per_channel_num_bits_list = [[self.model.num_bits * torch.ones(self.model.encoder.hidden_state_num_channels)]]
        autoencoder.exchange.save(exchange, 1, tensor_list, per_channel_num_bits_list, 0)


    def decode(self, exchange, output):

        # Load the .qtx file
        loaded_model_id, loaded_tensor_list = autoencoder.exchange.load(exchange)

        # Decode, apply transfer function and remap from [-1,1] to [0,1]
        y = self.model.decoder(loaded_tensor_list[0])
        y = torch.nn.functional.hardtanh(y)
        y = (y + 1) * 0.5

        # Build a PIL image out of the ouput tensor
        to_image = transforms.ToPILImage()
        image = to_image(y.reshape(y.shape[1], y.shape[2], y.shape[3]))

        image.save(output)


def encode(model, input, exchange):
    context = Context(model)
    context.encode(input, exchange)


def decode(model, exchange, output):
    context = Context(model)
    context.decode(exchange, output)