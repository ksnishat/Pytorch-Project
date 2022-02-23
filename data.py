from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

SPLIT_TRAIN = 0.8
SPLIT_VAL = 0.2


class ChallengeDataset(Dataset):
    def __init__(self, mode, csv_path, split, transform=tv.transforms.Compose([tv.transforms.ToTensor()])):
        self.mode = mode
        self.csv_path = csv_path
        self.data_frame = pd.read_csv(self.csv_path)
        self.transform = transform
        self.split = split

        if mode == 'train':
            self.train_data, self.test_data = train_test_split(self.data_frame, train_size=self.split, random_state=42)
        else:
            self.train_data, self.test_data = train_test_split(self.data_frame, test_size=self.split, random_state=42)

        self.train_labels = np.zeros((len(self.train_data), 2))
        for i in range(len(self.train_data)):
            # if i == 236:
            #     print(i)
            x = self.train_data.iloc[i, 0].split(';')[2:]
            self.train_labels[i] = x

        #print('constructor111')

    def __len__(self):
        #TODO change len according to mode
        x = None
        if self.mode == 'train':
            x = len(self.train_data)
        if self.mode == 'test':
            x = len(self.test_data)
        return x

    def __getitem__(self, index):
        x = self.train_data
        if self.mode == 'test':
            x = self.test_data

        img_data = x.iloc[index, 0].split(';')
        image_path = img_data[0]
        image_label = np.array(img_data[1:][1:], dtype=np.float32)
        raw_image = imread(image_path)
        rgb_image = gray2rgb(raw_image)
        tensor = rgb_image
        #tranform
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, image_label

    def safe_division(self, n, d):
        if d:
            return n/d
        else:
            return 0

    def pos_weight(self):
        x = np.sum(self.train_labels, axis=0)

        weight_crack = self.safe_division((self.__len__()-x[0]), x[0])
        weight_inactive = self.safe_division((self.__len__()-x[1]), x[1])

        return torch.tensor((weight_crack, weight_inactive))


def get_train_dataset():
    # TODO
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.RandomVerticalFlip(p=0.5),
                                       tv.transforms.RandomHorizontalFlip(p=0.5),
                                       tv.transforms.RandomAffine(20),
                                       tv.transforms.ColorJitter(brightness=0.2, contrast=0.7, saturation=0.2, hue=0.2),
                                       # tv.transforms.ColorJitter(hue=.05, saturation=.05),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize(train_mean, train_std)])

    return ChallengeDataset('train', './train.csv', SPLIT_TRAIN, transform)


# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    # TODO
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize(train_mean, train_std)])
    return ChallengeDataset('test', './train.csv', SPLIT_VAL, transform)

