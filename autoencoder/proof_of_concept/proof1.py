import torch
import torchvision
from autoencoder.tools.celeba import CelebA
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import autoencoder.transforms as transforms
import autoencoder.tools.datasets as datasets
import autoencoder.tools.show as show
import datetime


"""
This script trains a model with CelebA dataset. Copy this code as a template and change the AutoEncoder class to train your model.
A loss history figure is shown at the end and an image is passed through the model.

Set hparams['use_reduced_datasets'] to True to train with a fraction of the whole set.

Work in progress for improving this template. 
"""


hparams = {
    'batch_size': 32,
    'device': 'cuda',
    'max_dataset_size': 160,
    'log_interval': 2,
    'num_epochs': 40,
    'num_workers': 4,
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


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x




if __name__ == '__main__':
    # make CelebA dataset images available at celeba_image_folder
    celeba_root = os.path.join(os.path.expanduser('~'), 'autoencoder', 'datasets', 'celeba')
    celeba_image_folder = os.path.join(celeba_root, 'celeba', 'img_align_celeba')
    if not os.path.exists(celeba_image_folder):
        # celeba_image_folder is missing --> donwload CelebA
        Path(celeba_root).mkdir(parents=True, exist_ok=True)
        CelebA(celeba_root, download=True, transform=None)

    # transformations
    pre_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(160),
    ])

    pre_val_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(160),
    ])

    X_transform = torchvision.transforms.Compose([
        transforms.AddGaussianNoise(),
    ])

    post_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolderAutoEncoderDataset(
        celeba_image_folder,
        'train',
        pre_transform=pre_train_transform,
        X_transform=X_transform,
        post_transform=post_transform,
        max_size=hparams['max_dataset_size']
    )
    X, target = train_dataset[0]
    show.show_tensor_image(X)
    show.show_tensor_image(target)

    test_dataset = datasets.ImageFolderAutoEncoderDataset(
        celeba_image_folder,
        'test',
        pre_transform=pre_train_transform,
        X_transform=X_transform,
        post_transform=post_transform,
        max_size=hparams['max_dataset_size']
    )
    val_dataset = datasets.ImageFolderAutoEncoderDataset(
        celeba_image_folder,
        'val',
        pre_transform=pre_train_transform,
        X_transform=X_transform,
        post_transform=post_transform,
        max_size=hparams['max_dataset_size']
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=hparams['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=hparams['num_workers'])

    print("len(train_loader):", len(train_loader))
    print("len(test_loader):", len(test_loader))
    print("len(val_loader):", len(val_loader))

    iter_ = iter(train_loader)
    bX, btarget = next(iter_)
    print('Batch X shape: ', bX.shape)
    print('Batch target shape: ', btarget.shape)

    print('Output Batch Y shape: ', AutoEncoder()(bX).shape)

    model = AutoEncoder()
    model.to(hparams['device'])
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.MSELoss()

    tr_losses = []
    te_losses = []

    num_epochs = hparams['num_epochs']

    try:
        for epoch in range(1, num_epochs + 1):
            tr_loss = train_epoch(epoch, train_loader, model, optimizer, criterion, hparams)
            te_loss = eval_epoch(val_loader, model, criterion, hparams)
            tr_losses.append(tr_loss)
            te_losses.append(te_loss)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    plt.figure(figsize=(10, 8))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(tr_losses, label='train')
    plt.plot(te_losses, label='eval')
    plt.legend()
    plt.show()

    iter_ = iter(test_loader)
    bX, btarget = next(iter_)
    bX = bX[0, :, :, :]
    bX = bX.reshape(1, bX.shape[0], bX.shape[1], bX.shape[2])
    btarget = btarget[0, :, :, :]
    output = model(bX.to(hparams['device']))
    output = output.cpu()
    output = output.reshape(output.shape[1], output.shape[2], output.shape[3])
    show.show_tensor_image(btarget)
    show.show_tensor_image(bX.reshape(bX.shape[1], bX.shape[2], bX.shape[3]))
    show.show_tensor_image(output)
