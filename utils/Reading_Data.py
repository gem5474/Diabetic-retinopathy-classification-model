import pandas as pd
import random
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np


def read_split_data(root: str, val_rate: float = 0.2, plot_image: bool = False):
     # 保证随机结果可复现
    # random.seed(0)
    #确保路径正确
    assert os.path.exists(root), f'dataset root {root} does not exist.'
    #获取 excel 文件路径
    Retinopathy_classes = os.listdir(root)
    #file_path = 'D:\\DR\\Base11\\data_path_label.xls'
    #按照标签将图片进行分类
    image_path_0, image_path_1, image_path_2, image_path_3= [], [], [], []
    for i in Retinopathy_classes:
        data = pd.read_excel(root + '\\' + i)  #导入文件数据
        image_path = data['Image name']   #获取图片的文件名列表
        image_label = data['Retinopathy grade']   #获取标签列表

        for j in range(len(image_label)):
            image_lab = int(image_label[j])
            if image_lab == 0:
                image_path_0.append(image_path[j])
            elif image_lab == 1:
                image_path_1.append(image_path[j])
            elif image_lab == 2:
                image_path_2.append(image_path[j])
            elif image_lab == 3:
                image_path_3.append(image_path[j])

    #训练集所有图片的路径和对应索引信息
    train_images_path, train_images_label = [], []
    # 验证集所有图片的路径和对应索引信息
    val_images_path, val_images_label = [], []
    # 每个类别的样本总数
    every_class_num = []

    image_path_all = [image_path_0, image_path_1, image_path_2, image_path_3]

    for i, path in enumerate(image_path_all):
        '''
        i : 对应的是图片的类别标签 0, 1, 2, 3
        path : 路径列表, image_path_0, image_path_1, image_path_2, image_path_3
        '''
        # 获取此类别的样本数
        every_class_num.append(len(path))

        # 按比例随机采样验证集
        val_path = random.sample(path, k=int(len(path) * val_rate))

        for img_path in path:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(i)
            else:
                train_images_path.append(img_path)
                train_images_label.append(i)
    
    return train_images_path, train_images_label, val_images_path, val_images_label

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_label: list, allimages_path: str, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.allimages_path = allimages_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # img = Image.open(self.allimages_path + '\\' + self.images_path[item])
        img = cv2.imread(self.allimages_path + '\\' + self.images_path[item])
        randomflag = random.randint(1, 5)
        if randomflag == 1:
            # 垂直镜像
            img = cv2.flip(img, 1)
        elif randomflag == 2:
            img = cv2.flip(img, 0)
            # 水平垂直镜像
        elif randomflag == 3:
            # 水平垂直翻转
            img = cv2.flip(img, -1)
        elif randomflag == 4:
            # 90度旋转
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        img = Image.fromarray(np.uint8(img))
        if img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[item]} is not RGB mode")
        label = self.images_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

def get_dataset_dataloader(data_path, batch_size, allimages_path):
    train_images_path, train_iamges_label, val_images_path, val_images_label = read_split_data(root=data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     # transforms.Resize(512),
                                     # transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),

        "val": transforms.Compose([transforms.CenterCrop(224),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_label=train_iamges_label,
                              allimages_path=allimages_path,
                              transform=data_transform['train'])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_label=val_images_label,
                            allimages_path=allimages_path,
                            transform=data_transform['val'])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # nw = 1

    # print(f"Using {nw} dataloader workers every process.")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    return train_dataset, val_dataset, train_dataloader, val_dataloader
