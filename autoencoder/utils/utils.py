import torch
import torch.nn as nn
import autoencoder.datasets
import autoencoder.models
import autoencoder.utils
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import os
import math
import torch.optim as optim
import torchvision.utils
import autoencoder.datasets
import autoencoder.models
import autoencoder.utils


def create_dataloaders(hparams):
    # Set the seed to a known state for reproducibility of the training
    torch.manual_seed(0x1234)

    # Create datasets with all the train and test crops
    all_train_crops = autoencoder.datasets.CropsDataset(
        "./data/image_dataset/train",
        block_width=hparams['block_width'],
        block_height=hparams['block_height'])

    all_test_crops = autoencoder.datasets.CropsDataset(
        "./data/image_dataset/test",
        block_width=hparams['block_width'],
        block_height=hparams['block_height'])

    # Random split the original dataset in train, test and discarded datasets
    train_dataset_size = hparams['train_dataset_size']
    train_crops, _ = \
        torch.utils.data.random_split(all_train_crops, [train_dataset_size, len(all_train_crops) - train_dataset_size])

    test_dataset_size = hparams['test_dataset_size']
    test_crops, _ = \
        torch.utils.data.random_split(all_test_crops, [test_dataset_size, len(all_test_crops) - test_dataset_size])

    # Create the dataset transforms
    train_input_transform = transforms.Compose([transforms.ToTensor()])
    train_output_transform = transforms.Compose([transforms.ToTensor()])

    test_input_transform = transforms.Compose([transforms.ToTensor()])
    test_output_transform = transforms.Compose([transforms.ToTensor()])

    # Wrap the train and test samples with an XYDimsDataset
    train_xydims_samples = \
        autoencoder.datasets.XYDimsDataset(train_input_transform, train_output_transform, dataset=train_crops)

    test_xydims_samples = \
        autoencoder.datasets.XYDimsDataset(test_input_transform, test_output_transform, dataset=test_crops)

    # Create data loaders for train and test samples
    train_loader = torch.utils.data.DataLoader(train_xydims_samples, batch_size=hparams['batch_size'], shuffle=True,
                                               num_workers=hparams['num_workers'])

    test_loader = torch.utils.data.DataLoader(test_xydims_samples, batch_size=hparams['batch_size'], shuffle=False,
                                              num_workers=hparams['num_workers'])

    # Pick a few train samples
    few_train_x = [sample[0] for sample in [train_xydims_samples[index] for index in range(4)]]
    few_train_y = [sample[1] for sample in [train_xydims_samples[index] for index in range(4)]]

    # Move few_train_x to the same device where the inferences will be left
    for index in range(len(few_train_x)):
        few_train_x[index] = few_train_x[index].to(hparams['device'])

    # Move few_train_y to the same device where the inferences will be left
    for index in range(len(few_train_y)):
        few_train_y[index] = few_train_y[index].to(hparams['device'])

    # Pick a few test samples
    few_test_x = [sample[0] for sample in [test_xydims_samples[index] for index in range(4)]]
    few_test_y = [sample[1] for sample in [test_xydims_samples[index] for index in range(4)]]

    # Move few_test_x to the same device where the inferences will be left
    for index in range(len(few_test_x)):
        few_test_x[index] = few_test_x[index].to(hparams['device'])

    # Move few_test_y to the same device where the inferences will be left
    for index in range(len(few_test_y)):
        few_test_y[index] = few_test_y[index].to(hparams['device'])

    return train_loader, test_loader, few_train_x, few_train_y, few_test_x, few_test_y


# Some auxilary functions for the training loop
def train_epoch(train_loader, model, optimizer, criterion, hparams):
    np.random.seed(datetime.datetime.now().microsecond)
    model.train()
    device = hparams['device']
    losses = []
    for data, target, _, _ in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def test_epoch(test_loader, model, criterion, hparams):
    np.random.seed(0)
    model.eval()
    device = hparams['device']
    eval_losses = []
    with torch.no_grad():
        for data, target, _, _ in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            eval_losses.append(criterion(output, target).item())
    return np.mean(eval_losses)


def inference(model, inputs_list, hparams):
    """
    Do an inference with the model for each input tensor from the provided list and
    return three lists, one with the inference results, another with the differences
    in the [-1,1] range and a third one with the differences scaled up and shifted for
    visualizing then as images in the [0,1] range.
    """

    color_scale = 8.0

    outputs = []
    differences = []
    scaled_differences = []

    for x in inputs_list:

        num_channels = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]
        single_element_batch = x.clone().detach().reshape(1, num_channels, height, width)
        single_element_batch = single_element_batch.to(hparams['device'])
        model.to(hparams['device'])
        model.eval()

        output = model(single_element_batch)
        difference = single_element_batch - output
        scaled_difference = torch.clamp(difference * color_scale, min=-1.0, max=1.0) * 0.5  + 0.5
       
        output = output.reshape(num_channels, height, width)
        difference = difference.reshape(num_channels, height, width)
        scaled_difference = scaled_difference.reshape(num_channels, height, width)

        outputs.append(output)
        differences.append(difference)
        scaled_differences.append(scaled_difference)

    return outputs, differences, scaled_differences


def psnr(mean_square_normalized_error):
    max_i = 255.0
    mse = max(mean_square_normalized_error, 1e-10) * max_i * max_i
    return 20 * math.log10(max_i) - 10 * math.log10(mse)


def psnr2(mean_square_normalized_error):
    return - 10 * math.log10(max(mean_square_normalized_error, 1e-10))


def concat(list):
    result = None
    for tensor in list:
        if result is None:
            result = tensor
        else:
            result = torch.cat((result, tensor), dim=0)
    return result


def train(hparams, model, train_loader, test_loader, few_train_x, few_train_y, few_test_x, few_test_y):
    # Create the summary writer
    writer = SummaryWriter('share/' + hparams['name'] + '/runs')

    # Instantiate optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = nn.MSELoss()

    # Move model to device
    model = model.to(hparams['device'])

    # Checkpoint filename
    filename = 'share/' + hparams['name'] + "/model.pt"

    # Restore previous checkpoint or create new one from scratch
    if os.path.isfile(filename):
        print("Restoring from previous checkpoint")
        checkpoint = torch.load(filename)
    else:
        checkpoint = {
            'best_train_loss': None,
            'best_epoch': None,
            'best_model': None,
            'last_epoch': -1,
            'last_model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

    # Load model and optimizer from the checkpoint
    model.load_state_dict(checkpoint['last_model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Run a number of training epochs
    start = checkpoint['last_epoch'] + 1
    end = hparams['num_epochs']

    if start < end - 1 or checkpoint['best_train_loss'] is None:

        for epoch in range(start, end):

            train_loss = train_epoch(train_loader, model, optimizer, criterion, hparams)
            test_loss = test_epoch(test_loader, model, criterion, hparams)

            # Log losses
            writer.add_scalar("train_loss", train_loss, global_step=epoch)
            writer.add_scalar("test_loss", test_loss, global_step=epoch)

            # Log PSNRs
            train_psnr = psnr2(train_loss)
            test_psnr = psnr2(test_loss)
            writer.add_scalar("train_psnr", train_psnr, global_step=epoch)
            writer.add_scalar("test_psnr", test_psnr, global_step=epoch)

            if epoch == hparams['num_epochs'] - 1 or epoch % hparams['inference_freq'] == 0:

                # Show inferences with a few training samples,
                # one column per sample in (x, y, y_hat) format
                few_train_y_hat, few_train_diff, few_train_scaled_diff = inference(model, few_train_x, hparams)
                grid = torchvision.utils.make_grid(few_train_x + few_train_y + few_train_y_hat + few_train_scaled_diff, nrow=4)
                writer.add_image(tag='train', img_tensor=grid, global_step=epoch)

                # Show histogram of differences for a few training samples
                writer.add_histogram(tag="train_diff", values=concat(few_train_diff),  global_step=epoch)

                # Show inferences with a few test samples,
                # one column per sample in (x, y, y_hat) format
                few_test_y_hat, few_test_diff, few_test_scaled_diff  = inference(model, few_test_x, hparams)
                grid = torchvision.utils.make_grid(few_test_x + few_test_y + few_test_y_hat + few_test_scaled_diff, nrow=4)
                writer.add_image(tag='test', img_tensor=grid, global_step=epoch)

                # Show histogram of differences for a few training samples
                writer.add_histogram(tag="test_diff", values=concat(few_test_diff),  global_step=epoch)

                writer.flush()

            if checkpoint['best_train_loss'] is None or train_loss < checkpoint['best_train_loss']:

                # Update best model in the checkpoint
                checkpoint['best_train_loss'] = train_loss
                checkpoint['best_epoch'] = epoch
                checkpoint['best_model'] = model.state_dict()

            if epoch == hparams['num_epochs'] - 1 or epoch % hparams['checkpointing_freq'] == 0:

                print('Saving checkpoint at epoch ' + str(epoch))

                # Update last model and optimizer in the checkpoint
                checkpoint['last_epoch'] = epoch
                checkpoint['last_model'] = model.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()

                torch.save(checkpoint, filename)

    writer.close()
