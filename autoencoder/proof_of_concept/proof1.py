import torch
import torchvision
from autoencoder.tools.celeba import CelebA
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math
from pathlib import Path
import autoencoder.tools.datasets as datasets
import datetime
import autoencoder.tools.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import autoencoder.proof_of_concept.autoencoders as autoencoders
import autoencoder.proof_of_concept.transition as transition
import autoencoder.proof_of_concept.color as color
import autoencoder.proof_of_concept.resnet as resnet
import autoencoder.proof_of_concept.pyramidal2 as pyramidal


"""
This script trains a model with CelebA dataset. 
"""


hparams = {
    'batch_size': 1,
    'device': 'cuda',
    'max_dataset_size': 160,
    'log_interval': 2,
    'show_interval': 1,
    'num_epochs': 100,
    'num_workers': 4,
    'tensorboard_name': 'proof1',
}


def train_epoch(epoch, train_loader, model, optimizer, criterion, hparams):
    np.random.seed(datetime.datetime.now().microsecond)
    model.train()
    device = hparams['device']
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % hparams['log_interval'] == 0 or batch_idx >= len(train_loader):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def eval_epoch(val_loader, model, criterion, hparams):
    np.random.seed(0)
    model.eval()
    device = hparams['device']
    eval_losses = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            eval_losses.append(criterion(output, target).item())
    eval_loss = np.mean(eval_losses)
    print('Eval set: Average loss: {:.4f}'.format(eval_loss))
    return eval_loss


if __name__ == '__main__':
    # make CelebA dataset images available at celeba_image_folder
    celeba_root = os.path.join(os.path.expanduser('~'), 'autoencoder', 'datasets', 'celeba')
    celeba_image_folder = os.path.join(celeba_root, 'celeba', 'img_align_celeba')
    if not os.path.exists(celeba_image_folder):
        # celeba_image_folder is missing --> donwload CelebA
        Path(celeba_root).mkdir(parents=True, exist_ok=True)
        CelebA(celeba_root, download=True, transform=None)

    # transformations = pre + X + post

    # pre (train): randomness
    pre_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(160),
    ])

    # pre (val, test): no randomness
    pre_val_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(160),
    ])

    # X: noise, jpeg artifacts, B&W, etc
    X_transform = torchvision.transforms.Compose([
        transforms.AddGaussianNoise(),
    ])

    # post: to tensor from 0 to 1
    post_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # train dataset: the dataset returns (X, Y), where X is an image altered by X_transform above and Y is its corresponding ground truth
    train_dataset = datasets.ImageFolderAutoEncoderDataset(
        celeba_image_folder,
        'train',
        pre_transform=pre_train_transform,
        X_transform=X_transform,
        post_transform=post_transform,
        max_size=hparams['max_dataset_size']
    )
    X, target = train_dataset[0]

    # test dataset
    test_dataset = datasets.ImageFolderAutoEncoderDataset(
        celeba_image_folder,
        'test',
        pre_transform=pre_val_transform,
        X_transform=X_transform,
        post_transform=post_transform,
        max_size=hparams['max_dataset_size']
    )

    # validation dataset
    val_dataset = datasets.ImageFolderAutoEncoderDataset(
        celeba_image_folder,
        'val',
        pre_transform=pre_val_transform,
        X_transform=X_transform,
        post_transform=post_transform,
        max_size=hparams['max_dataset_size']
    )

    # data loaders: shuffle only train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=hparams['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=hparams['num_workers'])

    model = pyramidal.get_default()
    model = model.to(hparams['device'])
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.MSELoss()

    # init tensor board
    with SummaryWriter(os.path.join(os.path.expanduser('~'), 'autoencoder', 'tensorboard', hparams['tensorboard_name'], datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))) as writer:

        #model = autoencoders.Basic()
        #model = autoencoders.BasicFC(train_dataset[0][0].shape[1], train_dataset[0][0].shape[2], 0.2)
        #model = autoencoders.BasicNoDownscale()
        #model = autoencoders.Resnet14(48, resnet.ResidualBlockFactory(), transition.DownscaleConv2dFactory(), transition.UpscaleConvTranspose2dFactory())
        #model = autoencoders.Resnet14NoDownscale(48, resnet.ResidualBlockFactory())
        #model = autoencoders.Resnet14NoDownscale(48, resnet.FullPreactivationResidualBlockFactory())
        #model = autoencoders.Resnet14NoDownscale(8, resnet.ResidualBlockFactory())
        #model = autoencoders.Resnet28(32, resnet.ResidualBlockFactory(), transition.DownscaleAvgPool2dFactory(), transition.UpscalePixelShuffleFactory())


        try:
            for epoch in range(1, hparams['num_epochs'] + 1):
                # train and eval
                tr_loss = train_epoch(epoch, train_loader, model, optimizer, criterion, hparams)
                te_loss = eval_epoch(val_loader, model, criterion, hparams)

                # record losses
                writer.add_scalars('loss (MSE)', {
                    'train': tr_loss,
                    'validation': te_loss,
                }, epoch)
                writer.add_scalars('loss (PSNR)', {
                    'train': -10 * math.log10(tr_loss),
                    'validation': -10 * math.log10(te_loss),
                }, epoch)

                # record images
                np.random.seed(0)
                model.eval()
                with torch.no_grad():
                    # test
                    iter_ = iter(test_loader)
                    bX, btarget = next(iter_)
                    bX = bX[0:min(8, hparams['batch_size']), :, :, :]
                    btarget = btarget[0:min(8, hparams['batch_size']), :, :, :]
                    output = model(bX.to(hparams['device']))
                    writer.add_images('1. output (test)', output, epoch)

                    # input and ground truth
                    if epoch == 1:
                        writer.add_images('3. input', bX, epoch)
                        writer.add_images('4. ground truth', btarget, epoch)


                    # train
                    iter_ = iter(train_loader)
                    bX, btarget = next(iter_)
                    bX = bX[0:min(8, hparams['batch_size']), :, :, :]
                    btarget = btarget[0:min(8, hparams['batch_size']), :, :, :]
                    output = model(bX.to(hparams['device']))
                    writer.add_images('2. output (train)', output, epoch)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
