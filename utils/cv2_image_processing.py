import cv2
import numpy as np
from math import sqrt
import os
import argparse

def change_size(read_file):
    image = cv2.imread(read_file, 1)  # 读取图片 image_name应该是变量
    img = cv2.medianBlur(image, 3)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    x = binary_image.shape[0]
    # print("高度x=", x)
    y = binary_image.shape[1]
    # print("宽度y=", y)
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度
    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top - bottom  # 高度
    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    return pre1_picture  # 返回图片数据

# 锐化处理
def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    # cv2.imshow("custom_blur_demo", dst)
    return dst

# 彩图直方图均衡化
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
def custom_Equalize(image):

    (b, g, r) = cv2.split(image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return result

def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img1.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return dst

def contour_processing(image):
    rows, cols = image.shape[0: 2]
    row_new = int(rows / 2)
    col_new = int(cols / 2)
    hypotenuse = sqrt(rows ** 2 + cols ** 2)
    for i in np.arange(row_new, int(hypotenuse / 2)):
        img_new = cv2.circle(image, (row_new, col_new), i, (0, 0, 0),2)
    
    return img_new


def main(args):
    save_path = 'C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/transform_img'
    # file_names = 'D:\\Retinopathy\\images\\all_Images\\20051201_38262_0400_PP.tif'
    path = args.allimages_path  # 路径
    files = os.listdir(path)
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    for file in files:
        img_path = os.path.join(path, file)
        template = change_size(img_path)
        img = cv2.resize(template, (800, 800))
        img_2 = contour_processing(template)
        img_gaussianBlur = cv2.GaussianBlur(img_2, (3, 3), 0.4)
        # img_qua = custom_Equalize(img_gaussianBlur)
        img_qua = contrast_img(img_gaussianBlur, 0.5, 20)
        img_qua = custom_Equalize(img_qua)
        img_path = os.path.join(path, file)
        # data_enhance(img_path,save_path)

        cv2.imwrite(os.path.join(save_path + '\\' + str(os.path.basename(img_path))),
                    img_qua, ((int(cv2.IMWRITE_TIFF_RESUNIT), 1,
                                 int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
                                 int(cv2.IMWRITE_TIFF_XDPI), 1,
                                 int(cv2.IMWRITE_TIFF_YDPI), 1)))
    print('数据处理已完成')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default='C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/annotation')
    parser.add_argument('--flag', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allimages_path', type=str, default='C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/images')

    opt = parser.parse_args()
    # print(opt)
    main(opt)