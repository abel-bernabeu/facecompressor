import cv2
import numpy as np
import PIL


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=25.5):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        cv2_x = cv2_x.astype(np.float32)
        noise = np.random.normal(self.mean, self.std, cv2_x.shape)
        cv2_x = cv2_x + noise
        cv2_x = np.clip(cv2_x, 0, 255)
        cv2_x = np.round(cv2_x, decimals=0).astype(np.uint8)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_BGR2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x

class ConvertToGray(object):
    def __init__(self, mean=0., std=25.5):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        cv2_x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2GRAY)
        cv2_x = cv2.cvtColor(cv2_x, cv2.COLOR_GRAY2RGB)
        x = PIL.Image.fromarray(cv2_x)
        return x
