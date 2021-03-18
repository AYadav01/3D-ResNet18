import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise, img_as_ubyte
import random
import cv2
import torch

def make_binary(mask):
    coords = np.where(mask != 0)
    mask[coords] = 1
    return mask

class RandomRotate:
    def __call__(self, data):
        rotation_angle = random.randint(-180, 180)
        img, mask = data['img'], data['mask']
        img = rotate(img, rotation_angle, mode='reflect').astype(float)
        mask = make_binary(rotate(img_as_ubyte(mask), rotation_angle, mode='reflect')).astype(float)
        return {"img": img, "mask": mask}

class HorizontalFlip:
    def __call__(self, data, prob=0.5):
        if np.random.rand() < prob:
            img, mask = data['img'], data['mask']
            h_img = np.fliplr(img).astype(float)
            h_mask = np.fliplr(mask).astype(float)
            return {"img": h_img, "mask": h_mask}
        else:
            return data

class VerticalFlip:
    def __call__(self, data, prob=0.5):
        if np.random.rand() < prob:
            img, mask = data['img'], data['mask']
            v_img = np.flipud(img).astype(float)
            v_mask = np.flipud(mask).astype(float)
            return {"img": v_img, "mask": v_mask}
        else:
            return data

class RandomNoise:
    def __call__(self, data, prob=0.5):
        if np.random.rand() < prob:
            img, mask = data['img'], data['mask']
            noised_img = random_noise(img).astype(float)
            mask = mask.astype(float)
            return {"img": noised_img, "mask": mask}
        else:
            return data

class RandomBlur:
    def __call__(self, data):
        img, mask = data['img'], data['mask']
        blur_factor = random.randrange(1, 10, 2)
        blurred_img = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
        return {"img": blurred_img, "mask": mask}

class ToTensor:
    def __call__(self, data):
        img, mask = data['img'], data['mask']
        tensored_img = torch.from_numpy(img.transpose((2,0,1))).type(torch.FloatTensor)
        tensored_mask = torch.from_numpy(np.expand_dims(mask, axis=2).transpose((2,0,1))).type(torch.FloatTensor)
        return {"img": tensored_img, "mask": tensored_mask}

