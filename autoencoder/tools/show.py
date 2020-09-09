import matplotlib.pyplot as plt
import numpy as np


def show_pil_image(pil_im):
    """Shows PIL image made of bytes from 0 to 255"""
    plt.imshow(np.asarray(pil_im))
    plt.show()

def show_tensor_image(tensor_im):
    """Shows tensor image made of float32s from 0 to 1, C x H x W"""
    tensor_im = tensor_im.permute(1, 2, 0)
    pil_im = np.clip(tensor_im.detach().numpy(), 0, 1)
    pil_im = np.round(pil_im * 255, 0)
    pil_im = pil_im.astype(np.uint8)
    show_pil_image(pil_im)
