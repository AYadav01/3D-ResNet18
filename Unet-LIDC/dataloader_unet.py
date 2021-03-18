from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from skimage.color import gray2rgb
import pickle

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, masks_dir, transformation=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transformations = transformation
        self.imgs_ids = [file for file in listdir(imgs_dir)]
        self.mask_ids = [file for file in listdir(masks_dir)]

    def _check_for_mask(self, mask):
        if len(np.unique(mask)) != 2:
            coords = np.where(mask != 0)
            mask[coords] = 1
        return mask

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        mask_idx = self.mask_ids[i]
        image_path = self.imgs_dir + img_idx
        mask_path = self.masks_dir + mask_idx
        # Open pickle file
        with open(image_path, "br") as file:
            image = pickle.load(file)
            image = (image - image.min()) / (image.max() - image.min())
            image = gray2rgb(image)
        with open(mask_path, "br") as file:
            mask = pickle.load(file)
            mask = self._check_for_mask(mask)

        # fig, axs = plt.subplots(2)
        # axs[0].imshow(image, cmap="gray")
        # axs[1].imshow(mask, cmap="gray")
        # plt.show()
        # ----------------------------------------------
        # print(image_path.shape, mask_path.shape)
        # print("image min-max before conversion:", image_path.min(), image_path.max())
        # plt.imshow(image_path, cmap="gray")
        # plt.show()
        # -----------------------------------------------
        # image_path = (image_path - image_path.min()) / (image_path.max() - image_path.min())
        # print("image min-max after rescaling:", rescaled.min(), rescaled.max())
        # plt.imshow(rescaled, cmap="gray")
        # plt.show()
        # -------------------------------------------------
        # image = Image.fromarray(image).convert("RGB")
        # mask = Image.fromarray(mask).convert('L')
        data = {"img":image, "mask":mask}
        if self.transformations is not None:
            data = self.transformations(data)
        return data

    def __len__(self):
        return len(self.imgs_ids)


