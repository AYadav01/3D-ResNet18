import torch
import numpy as np
import os
from torch.utils.data import Dataset
import sys
import csv
from torch.utils.data import DataLoader
import pickle

class HscnnDataset(Dataset):
    def __init__(self, path_to_data, path_to_label_csv, transform=None):
        """
        Args:
            path_to_data (string): Path to images (.npy format supported)
            path_to_label_csv (string): CSV file with training/validation labels
            num_low_level_tasks (int, optional): number of low level tasks for the model
        """
        self.parsed_data = self._load_pickle(path_to_data)
        self.path_to_label_csv = path_to_label_csv
        self.transform = transform
        try:
            with self._open_for_csv(self.path_to_label_csv) as file:
                # Dictionary with keys as file name and values as labels
                self.image_data = self._read_labels(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV file: {}: {}'.format(self.path_to_data, e)), None)
        self.image_names = list(self.image_data.keys())

    def _load_pickle(self, path_to_data):
        """
        Reads a pickle file and return numpy array
        """
        with open(path_to_data, "rb") as f:
            file = pickle._Unpickler(f)
            file.encoding = 'latin1'
            file = file.load()
            return file

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader. For python2, open with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def __len__(self):
        """
        :return: Number of samples in dataset
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        :param idx: index value of the image currently in memory
        :return: dictionary with image, label
        """
        img = self.load_image(idx, add_dim=True)
        label = self.load_labels(idx)
        if self.transform:
            img_transform = self.transform(img)
        sample = {'img': img, 'label': label}
        return sample

    def load_image(self, image_index, upper_bound=-500, lower_bound=-1000, add_dim=False):
        """
        :param image_index: index value of the image currently in memory
        :param upper_bound: upper bound for intensity values
        :param lower_bound: lower bound for intensity values
        :param add_dim: adds a channel dimension to the image
        :return: limits the image values between the two bounds and scales the image betwen 0-1
        """
        #path_to_img = os.path.join(self.path_to_data, self.image_names[image_index])
        img = self.parsed_data[image_index].astype(np.float32)
        img[img > upper_bound] = upper_bound
        img[img < lower_bound] = lower_bound
        #array = (img - lower_bound) / (upper_bound - lower_bound)
        array = (img - img.min()) / (img.max() - img.min())
        if add_dim:
            array = np.expand_dims(array, axis=3)
            array = array.transpose((3, 0, 1, 2))
        return torch.from_numpy(array)

    def load_labels(self, image_index):
        """
        :param image_index: index value of the image currently in memory
        :return: labels in tensor format
        """
        label_list = self.image_data[self.image_names[image_index]]
        if len(label_list) > 1:
            raise (ValueError('Multiple label found for image {}:'.format(self.image_names[image_index])), None)
        else:
            data = [int(arg) for arg in list(label_list[0].values())]
            malignancy_lbl = data[-1]
            tensor_lbl = torch.from_numpy(np.array(malignancy_lbl))
        return tensor_lbl

    def _read_labels(self, csv_reader, header_available=True):
        """
        Returns a dictionary with index number as key and labels as values
        """
        if header_available:
            next(csv_reader)
        result = {}
        for line, col in enumerate(csv_reader):
            label_dict = {}
            try:
                img_file = str(line)
                if img_file not in result:
                    result[img_file] = []
                for i in range(len(col)):
                    label_dict['label'+str(i+1)] = col[i]
                result[img_file].append(label_dict)
            except ValueError:
                raise(ValueError(
                    'line {}: format should be \'label1,label2,label3,label4, img_file\' or \',,,,img_file\''.format(
                        line)), None)
        return result