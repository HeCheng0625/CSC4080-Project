import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

BATCH_SIZE = 32

def dataset_process(csv_file, data_augment=False):
    df = pd.read_csv(csv_file)
    id_label_list = []
    for i in range(len(df['id_code'])):
        id_label_list.append((df['id_code'][i], df['diagnosis'][i]))
    train_list, test_list =  train_test_split(id_label_list, random_state=42, train_size=0.8)
    train_list, test_list = train_list[:1600], test_list[:400]
    # data augmentation
    if data_augment:
        for i in range(1600):
            id, label = train_list[i]
            if (label == 1):
                train_list.append((id, label))
                train_list.append((id, label))
            if (label == 3):
                train_list.append((id, label))
                train_list.append((id, label))
                train_list.append((id, label))
            if (label == 4):
                train_list.append((id, label))
                train_list.append((id, label))
    return train_list, test_list
    # train_sta = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    # test_sta = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    # for i in range(len(train_list)):
    #     _, label = train_list[i]
    #     train_sta[label] += 1
    # for i in range(len(test_list)):
    #     _, label = test_list[i]
    #     test_sta[label] += 1

    # for k, v in train_sta.items():
    #     train_sta[k] = train_sta[k]/len(train_list)
    # for k, v in test_sta.items():
    #     test_sta[k] = test_sta[k]/len(test_list)

    # print(train_sta, len(train_list))
    # print(test_sta, len(test_list))

# dataset_process('train.csv', data_augment=True)

class MyDataset(Dataset):
    def __init__(self, data_dir, csv_file, train=True, transform=None, data_augment=False, augment_transforms=None):
        super(MyDataset, self).__init__()
        self.train_list, self.test_list = dataset_process(csv_file, data_augment)
        self.data_info = []
        if train:
            for i in range(len(self.train_list)):
                id, label = self.train_list[i]
                self.data_info.append((data_dir+'/'+id+'.png', label))
        else:
            for i in range(len(self.test_list)):
                id, label = self.test_list[i]
                self.data_info.append((data_dir+'/'+id+'.png', label))

        self.data_augment = data_augment
        self.transform = transform
        self.augment_transforms = augment_transforms

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if not self.data_augment:
            if self.transform is not None:
                img = self.transform(img)
        else:
            if index < 1600:
                if self.transform is not None:
                    img = self.transform(img)
            else:
                if self.augment_transforms is not None:
                    img = self.augment_transforms[index%len(self.augment_transforms)](img)
                else:
                    if self.transform is not None:
                        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)