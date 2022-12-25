from data_processing.Dataset import Dataset
from data_processing.ImageTransform import ImageTransform
from lib import *
from constant import *


class Load_data:
    def __init__(self):
        pass

    def make_datapath_list(self, phase='train', data_path = DATA_PATH):
        target_path = os.path.join(data_path, phase, "**\*")
        path_list = []
        for path in glob.glob(target_path):
            path_list.append(path)

        return path_list

    def __call__(self, phase='train', batch_size=BATCH_SIZE):
        list_data = self.make_datapath_list(phase=phase)
        # print(len(list_data))
        # remove wrong image
        path = []
        for img_path in list_data:
            if "imageNotFound" in img_path:
                # print(img_path)
                path.append(img_path)

        for i in range(len(path)):
            list_data.remove(path[i])

        dataset = Dataset(list_data, transform=ImageTransform(), phase=phase)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
