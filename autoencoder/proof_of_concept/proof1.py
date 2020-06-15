import torch
import torchvision
import autoencoder.tools.celeba as celeba
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


"""
This script trains a model with CelebA dataset. Copy this code as a template and change the AutoEncoder class to train your model.
A loss history figure is shown at the end and an image is passed through the model.

Set hparams['use_reduced_datasets'] to True to train with a fraction of the whole set.

Work in progress for improving this template. 
"""


hparams = {
    'batch_size': 128,
    'device': 'cuda',
    'use_reduced_datasets': True,
    'log_interval': 2,
    'num_epochs': 3,
}

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    #torchvision.transforms.RandomRotation(10),
    #torchvision.transforms.RandomResizedCrop(160, scale=(0.09, 1.0), ratio=(0.99, 1.01)),
    torchvision.transforms.RandomCrop(160),
    torchvision.transforms.ToTensor(),
])

transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(160),
    torchvision.transforms.ToTensor()
])


def show_pil_image(pil_im):
    plt.imshow(np.asarray(pil_im))
    plt.show()

def show_tensor_image(tensor_im):
    tensor_im = tensor_im.permute(1, 2, 0)
    pil_im = np.clip(tensor_im.detach().numpy(), 0, 1)
    pil_im = np.round(pil_im * 255, 0)
    pil_im = pil_im.astype(np.uint8)
    show_pil_image(pil_im)


def train_epoch(epoch, train_loader, model, optimizer, criterion, hparams):
    model.train()
    device = hparams['device']
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % hparams['log_interval'] == 0 or batch_idx >= len(train_loader):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)

def eval_epoch(val_loader, model, criterion, hparams):
    model.eval()
    device = hparams['device']
    eval_losses = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            eval_losses.append(criterion(output, data).item())
    eval_loss = np.mean(eval_losses)
    print('Eval set: Average loss: {:.4f}'.format(eval_loss))
    return eval_loss

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x


if __name__ == '__main__':
    # save celeba dataset is ~/autoencoder/datasets/celeba folder
    celeba_dir = os.path.join(os.path.expanduser('~'), 'autoencoder', 'datasets', 'celeba')
    Path(celeba_dir).mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, val_loader = celeba.get_celeba_loaders(
        celeba_dir,
        hparams['batch_size'],
        train_transforms,
        transforms,
        transforms,
        use_reduced_datasets=hparams['use_reduced_datasets']
    )

    print("len(train_loader):", len(train_loader))
    print("len(test_loader):", len(test_loader))
    print("len(val_loader):", len(val_loader))

    iter_ = iter(train_loader)
    bimg, blabel = next(iter_)
    print('Batch Img shape: ', bimg.shape)
    print('Batch Label shape: ', blabel.shape)
    print('The Batched tensors return a collection of {} images ({} channel, {} height pixels, {} width pixels)'.format(bimg.shape[0], bimg.shape[1], bimg.shape[2], bimg.shape[3]))

    print('Output Batch Img shape: ', AutoEncoder()(bimg).shape)

    from PIL import Image

    model = AutoEncoder()
    model.to(hparams['device'])
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.MSELoss()

    tr_losses = []
    te_losses = []

    num_epochs = hparams['num_epochs']

    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(epoch, train_loader, model, optimizer, criterion, hparams)
        te_loss = eval_epoch(val_loader, model, criterion, hparams)
        tr_losses.append(tr_loss)
        te_losses.append(te_loss)

    plt.figure(figsize=(10, 8))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(tr_losses, label='train')
    plt.plot(te_losses, label='eval')
    plt.legend()
    plt.show()

    from PIL import Image

    pil_image = Image.open(os.path.join(celeba_dir, "celeba", "img_align_celeba", "000054.jpg"))
    show_pil_image(pil_image)
    tensor_image = transforms(pil_image)
    tensor_image = tensor_image.reshape([1, 3, 160, 160])
    tensor_image = tensor_image.to(hparams['device'])
    output = model(tensor_image).cpu()
    output = output.reshape([3, 160, 160])
    show_tensor_image(output)