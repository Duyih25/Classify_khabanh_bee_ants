from lib import *

class Dataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, indx):
        img_path = self.file_list[indx]
        img = Image.open(img_path)

        img = self.transform(img, self.phase)

        label = img_path.split('\\')[-2]
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1
        elif label == 'khabanh':
            label = 2

        return img, label
