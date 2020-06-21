from torch.utils.data.dataset import Dataset
import os
import PIL


class ImageFolderAutoEncoderDataset(Dataset):
    """This class creates a Dataset with all the images contained inside a given folder. The Dataset return tuples of X, target: X = image in the folder,
    target = ground truth of that image. So this type of Dataset is oriented to autoencoders.

    The transformations are applied as follows:

        X:          pre_transform + X_transform + post_transform
        target:     pre_transform + post_transform

    root: the folder where the images are contained
    split: train, test or val
    train_percentage: percentage of the images that are used for training. The remaining ones are splitted 50/50 between test and val
    pre_transform: RandomHorizontalFlip(), RandomCrop(160), etc
    X_transform: AddGaussianNoise(), ConvertToGray(), etc
    post_transform: ToTensor()
    """

    def __init__(self, root, split="train", train_percentage=0.8, pre_transform=None, X_transform=None, post_transform=None, max_size=None):
        super().__init__()

        self.root = root
        self.split = split
        self.train_percentage = train_percentage
        self.pre_transform = pre_transform
        self.X_transform = X_transform
        self.post_transform = post_transform
        self.max_size = max_size

        self.instances = []

        for root, _, fnames in sorted(os.walk(self.root, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(self.root, fname)
                self.instances.append(path)

        if max_size is not None:
            self.instances = self.instances[0:max_size]

        test_first = int(len(self.instances) * train_percentage)
        val_first = (len(self.instances) + test_first) // 2

        if split == 'train':
            self.instances = self.instances[0:test_first]
        elif split == 'test':
            self.instances = self.instances[test_first:val_first]
        elif split == 'val':
            self.instances = self.instances[val_first:]
        else:
            raise ValueError(f"split value not supported: {split}")

    def __getitem__(self, index):
        X = PIL.Image.open(self.instances[index])

        if self.pre_transform is not None:
            X = self.pre_transform(X)

        target = X

        if self.X_transform is not None:
            X = self.X_transform(X)

        if self.post_transform is not None:
            X = self.post_transform(X)
            target = self.post_transform(target)

        return X, target

    def __len__(self):
        return len(self.instances)
