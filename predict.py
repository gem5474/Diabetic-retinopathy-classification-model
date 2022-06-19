# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : predict.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import pandas as pd
import argparse


import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.base_model import BaseModel
import xlsxwriter

def ResultOutput(filename,excle_path,bodytitle,PhotoName,reallabel,prelabel,row):
    '''
    :param filename:文件名
    :param excle_path:保存路径
    :param bodytitle: 表头
    :param PhotoName: 图片名
    :param reallabel: 真实标签
    :param prelabel: 预测标签
    '''

    # excel路径
    excle_path = excle_path
    # 创建一个Workbook模块
    data = xlsxwriter.Workbook(filename)
    # 创建一个表格，cell_overwrite_ok=True 为不覆盖表，默认为False
    worksheet = data.add_worksheet()
    for i in range(0, len(bodytitle)):
        worksheet.write(0, i, bodytitle[i])
    print(range(2))
    # for i in range(len(PhotoName)):
    #     for j in range(2):
    #         if j == 0:
    #             worksheet.write(i + 1, j, PhotoName[i])
    #         elif j == 1:
    #             worksheet.write(i + 1, j, reallabel[i])
    #         elif j == 2:
    #             worksheet.write(i + 1, j, prelabel[i])
    for i in range(len(reallabel)):
                worksheet.write(i + 1+row, 0, PhotoName[i+row])
                worksheet.write(i + 1+row, 1, reallabel[i])
                worksheet.write(i + 1+row, 2, prelabel[i])

    # 保存到excel中
    # data.save(excle_path)
    data.close()
def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = args.allimages_path  # 路径
    image_path = []
    image_label = []
    prelabel = []
    files = os.listdir(path)
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。

    # 获取 excel 文件路径
    Retinopathy_classes = os.listdir(args.data_path)
    # 获取数据集名称和列表

    for file in files:
        data_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        img_path = os.path.join(path, file)
        assert os.path.exists(img_path), f"file {img_path} dose not exist."
        img = Image.open(img_path)
        # plt.imshow(img)
        img = data_transform(img)
        # [C, H, W] -> [1, C, H, W]
        img = torch.unsqueeze(img, dim=0)

        model = BaseModel(name=args.model_name, num_classes=args.num_classes).to(device)

        model.load_state_dict(torch.load(args.model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            # print(str(predict_cla))
            prelabel.append(str(predict_cla))
            if len(prelabel) <= 5:
                print(prelabel)
    print('完成预测，进行数据写入')
    # prelabel是所有预测的结果
    count = 0
    for i in Retinopathy_classes:
        count += 1
        print('当前循环数为')
        print(count)
        data = pd.read_excel(args.data_path + '\\' + i)  #导入文件数据
        image_path = data['Image name']   #获取图片的文件名列表
        image_label = data['Retinopathy grade']   #获取标签列表
        excle_path = 'C:/Users/17809/Desktop/Yuxi_test1'
        bodytitle = ['Image name', 'Retinopathy grade', 'Predict grade']
        image_path = image_path
        image_label = image_label
        filename = 'ResnetPredictResult.xlsx'
        ResultOutput(filename=filename,
                     excle_path=excle_path,
                     bodytitle=bodytitle,
                     PhotoName=image_path,
                     reallabel=image_label,
                     prelabel=prelabel,
                     row=count*len(image_path))
    print('预测结束')


#预测数据测试集 ’C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/pretest/annotation‘

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/annotation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allimages_path', type=str,
                        default='C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/transform_img')
    parser.add_argument('--model_weight_path', type=str, default='./weights/resnet.pth')
    args = parser.parse_args()
    main(args)