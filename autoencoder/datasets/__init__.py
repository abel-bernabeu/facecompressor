import torch.utils.data as data
import torchvision.transforms.functional as functional
import os
from PIL import Image


class CropsDataset(data.Dataset):
    """
    This dataset class reads all the images from a given directory and produces
    crops of the requested size.

    When the image dimensions are not a multiple of the requested block size,
    the image reflection is added as padding on the right and bottom edges.

    If no cropping is needed and all the images in the dataset are the same
    size then you can just crop to the given image size, which has a neutral
    effect because each crop gives you a full image.

    More especifically, when loading images from VoxCeleb1 which are 224x224,
    if no cropping is required by the model then we just do 224x224 crops which
    returns whole images.

    Example:

    tiles = CropsDataset("./images", 400, 200)

    images = []
    for tile, width, height in tiles:
        images.append(tile)
        print(width, height)

    fig = plt.figure(figsize=(100., 100.))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(4, 3),
                     axes_pad=0.1,
                     )

    for ax, im in zip(grid, images):
        ax.imshow(im)

    plt.show()
    """

    def __init__(self, directory, block_width, block_height, train=True, transform=None, download=False):

        super(CropsDataset).__init__()

        self.block_width = block_width
        self.block_height = block_height
        self.images = []
        self.original_widths = []
        self.original_heights = []
        self.image_for_index = []
        self.tile_for_index = []

        for _, _, filenames in os.walk(directory):
            for filename in filenames:

                filename = os.path.join(directory, filename)

                image = Image.open(filename)

                # Save the dimensions before any padding
                self.original_widths.append(image.width)
                self.original_heights.append(image.height)

                # If needed pad with reflection on right and bottom edges
                left = 0
                top = 0

                if image.width % block_width > 0:
                    right = block_width - image.width % block_width
                else:
                    right = 0

                if image.height % block_height > 0:
                    bottom = block_height - image.height % block_height
                else:
                    borrom = 0

                image = functional.pad(image, padding=(left, top, right, bottom), fill=0, padding_mode='reflect')

                self.images.append(image)
                height_blocks = self.get_height_blocks(image)
                width_blocks = self.get_width_blocks(image)

                image_tile_index = 0
                for row_block in range(height_blocks):
                    for col_block in range(width_blocks):
                        self.image_for_index.append(len(self.images) - 1)
                        self.tile_for_index.append(image_tile_index)
                        image_tile_index += 1

    def get_width_blocks(self, image):
        return (image.width + self.block_width - 1) // self.block_width

    def get_height_blocks(self, image):
        return (image.height + self.block_height - 1) // self.block_height

    def __len__(self):
        return len(self.tile_image)

    def __getitem__(self, index):
        """
        Returns a triplet with crop, width and height. The width and height will
        typically be self.block_width and self.block_height, but for blocks on
        the right and bottom edges there may be padding, which makes the effective
        size to be smaller.

        Some models or errror metrics may be capable of ignoring the padding,
        rather than learning to reproduce the padding. Those models can use the
        returned crop width and height.
        """
        image_index = self.image_for_index[index]
        image = self.images[image_index]
        original_width = self.original_widths[image_index]
        original_height = self.original_heights[image_index]
        index = self.tile_for_index[index]

        block_column_index = index % self.get_width_blocks(image)
        block_row_index = index // self.get_width_blocks(image)
        left = block_column_index * self.block_width
        top = block_row_index * self.block_height

        crop = functional.crop(image, top, left, self.block_height, self.block_width)

        if left + self.block_width > original_width:
            width = original_width % self.block_width
        else:
            width = self.block_width

        if top + self.block_height > original_height:
            height = original_height % self.block_height
        else:
            height = self.block_height

        return crop, width, height