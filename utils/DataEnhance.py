import numpy as np
import os
import cv2
import argparse


def data_enhance(root: str, save_path:str, gammarate: float = 0.8, Cbdflag:bool = 1):
    '''
    数据集处理，锐化提高对比度，直方化提高对比度，gamma变换调低亮度，
    root: str  图片路径
    save_path:str   增强后保存路径
    gamma: float = 0.8  gamma值
    flag:bool = 1  开关
    '''

    # 数据集处理，直方化提高对比度，gamma变换调低亮度，锐化提高对比度

    img_path = root
    gamma = gammarate
    image_path_save = save_path

    # img_path = "C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/text1/20051019_38557_0100_PP.tif"
    # image_path_save = "C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/transform_img"

    img = cv2.imread(img_path, -1)

    Cbdflag = Cbdflag
    if Cbdflag == 1:
        # 锐化处理
        def custom_blur_demo(image):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
            dst = cv2.filter2D(image, -1, kernel=kernel)
            # cv2.imshow("custom_blur_demo", dst)
            return dst

        # 彩图直方图均衡化
        # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
        def custom_Equalize(image):

            (b, g, r) = cv2.split(img)
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

        src = img
        img_Cbd = custom_blur_demo(src)
        img_qua = custom_Equalize(img_Cbd)
        img_gamma = adjust_gamma(img_qua, gamma)


        cv2.imwrite(os.path.join(image_path_save + '\\' + str(os.path.basename(img_path)) ),
                    img_gamma, ((int(cv2.IMWRITE_TIFF_RESUNIT), 1,
                                                              int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
                                                              int(cv2.IMWRITE_TIFF_XDPI), 1,
                                                              int(cv2.IMWRITE_TIFF_YDPI), 1)))
        return img_gamma


def main(args):
    save_path = 'C:/Users/17809/Desktop/Yuxi_test1/Retinopathy/transform_img'

    path = args.allimages_path  # 路径
    files = os.listdir(path)
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    for file in files:
        img_path = os.path.join(path, file)
        data_enhance(img_path,save_path)
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
    print(opt)
    main(opt)
