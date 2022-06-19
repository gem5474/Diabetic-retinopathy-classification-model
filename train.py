# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import math
import argparse
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter

from utils.Reading_Data import get_dataset_dataloader
from utils.train_val_utils import train_one_epoch, evaluate
from models.base_model import BaseModel
import matplotlib.pyplot as plt

import queue
from ModelPerformance import modelPerformance


def main(args):
    # 选中显卡来跑
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    print(torch.cuda.is_available())
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter('./logs')


    # 获取模型
    model = BaseModel(name=args.model_name, num_classes=args.num_classes).to(device)

    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5E-5)

    # cosine
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0

    acc_list = []
    bestacc_list = []
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f"Using {nw} dataloader workers every process.")
    start = time.time()
    for epoch in range(args.epochs):
        # 获取数据集

        train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataset_dataloader(args.data_path,
                                                                                              args.batch_size,
                                                                                              args.allimages_path)

        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch
        )

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch
        )

        # tensorboard
        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/" + args.model_name + ".pth")
        acc_list.append(val_acc)
        bestacc_list.append(best_acc)

    end = time.time()
    print("Training 耗时为:{:.1f}".format(end - start))
    print("The best_acc is %f" %(best_acc))

    # modelPerformance(name, y_true, y_pred)

    # ax1 = plt.subplot(221)
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.plot(acc_list, color='darkorange', label='(1+5%)$W_0$')
    plt.plot(bestacc_list, color='deepskyblue', label='X')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='densenet')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default='C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/annotation')
    parser.add_argument('--flag', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allimages_path', type=str,
                        default='C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/transform_img')
    # parser.add_argument('--allimages_path', type=str,default='C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/images')

    opt = parser.parse_args()
    print(opt)
    main(opt)