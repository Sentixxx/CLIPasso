import cv2
import os
import numpy as np
import re

def create_video_from_two_folders(folder1, folder2, output_video_path, fps=30):
    # 检查文件夹路径是否存在
    for folder in [folder1, folder2]:
        if not os.path.exists(folder):
            print(f"文件夹 {folder} 不存在！")
            return

    # 获取每个文件夹中的图片文件，并按文件名iter{num}.jpg排序 
    def sort_key(filename):
        match = re.search(r'iter(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    def sort_key2(filename):
        match = re.search(r'iter_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    images1 = sorted([img for img in os.listdir(folder1) if img.endswith(('jpg', 'jpeg', 'png'))], key=sort_key2)
    images2 = sorted([img for img in os.listdir(folder2) if img.endswith(('jpg', 'jpeg', 'png'))], key=sort_key)

    # 检查每个文件夹是否有相同数量的图片
    if len(images1) != len(images2):
        print("每个文件夹中的图片数量不相同！")
        return

    # 读取第一张图片来确定视频的尺寸
    first_image1 = cv2.imread(os.path.join(folder1, images1[0]))
    height, width, _ = first_image1.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))  # 视频尺寸为两张图片水平拼接

    # 逐一拼接每对图片并写入视频
    for i in range(len(images1)):
        # 读取每个文件夹中的图片
        image1 = cv2.imread(os.path.join(folder1, images1[i]))
        image2 = cv2.imread(os.path.join(folder2, images2[i]))
        print(images1[i], images2[i])

        # 缩放每个图像，使其适应目标画布的各个部分
        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))

        # 创建一个大画布 (1x2拼接，即左右拼接)
        canvas = 255 * np.ones(shape=(height, width * 2, 3), dtype=np.uint8)  # 白色背景

        # 将两张图片分别放到画布的左右部分
        canvas[0:height, 0:width] = image1  # 左边部分
        canvas[0:height, width:width*2] = image2  # 右边部分

        # 将拼接后的画面写入视频
        video_writer.write(canvas)

    # 释放资源
    video_writer.release()
    print(f"视频已成功创建：{output_video_path}")

def create_video_from_four_folders(folder1, folder2, folder3, folder4, output_video_path, fps=30):
    # 检查文件夹路径是否存在
    for folder in [folder1, folder2, folder3, folder4]:
        if not os.path.exists(folder):
            print(f"文件夹 {folder} 不存在！")
            return

    # 获取每个文件夹中的图片文件，并按文件名排序
    images1 = sorted([img for img in os.listdir(folder1) if img.endswith(('jpg', 'jpeg', 'png'))])
    images2 = sorted([img for img in os.listdir(folder2) if img.endswith(('jpg', 'jpeg', 'png'))])
    images3 = sorted([img for img in os.listdir(folder3) if img.endswith(('jpg', 'jpeg', 'png'))])
    images4 = sorted([img for img in os.listdir(folder4) if img.endswith(('jpg', 'jpeg', 'png'))])

    # 检查每个文件夹是否有相同数量的图片
    if len(images1) != len(images2) or len(images1) != len(images3) or len(images1) != len(images4):
        print("每个文件夹中的图片数量不相同！")
        return

    # 读取第一张图片来确定视频的尺寸
    first_image1 = cv2.imread(os.path.join(folder1, images1[0]))
    height, width, _ = first_image1.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height * 2))  # 视频尺寸为四个角拼接后的尺寸

    # 逐一拼接每个文件夹的图片并写入视频
    for i in range(len(images1)):
        # 读取每个文件夹中的图片
        image1 = cv2.imread(os.path.join(folder1, images1[i]))
        image2 = cv2.imread(os.path.join(folder2, images2[i]))
        image3 = cv2.imread(os.path.join(folder3, images3[i]))
        image4 = cv2.imread(os.path.join(folder4, images4[i]))

        # 缩放每个图像，使其适应目标画布的各个部分
        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))
        image3 = cv2.resize(image3, (width, height))
        image4 = cv2.resize(image4, (width, height))

        # 创建一个大画布 (2x2拼接)
        canvas = 255 * np.ones(shape=(height * 2, width * 2, 3), dtype=np.uint8)  # 白色背景

        # 将四个文件夹中的图片放到画布的四个角落
        canvas[0:height, 0:width] = image1  # 左上角
        canvas[0:height, width:width*2] = image2  # 右上角
        canvas[height:height*2, 0:width] = image3  # 左下角
        canvas[height:height*2, width:width*2] = image4  # 右下角

        # 将拼接后的画面写入视频
        video_writer.write(canvas)

    # 释放资源
    video_writer.release()
    print(f"视频已成功创建：{output_video_path}")

def create_video_from_four_folders1(folder1, folder2, folder3, folder4, output_video_path, fps=30):
    # 获取每个文件夹中的图片文件，并按文件名排序
    images1 = sorted([img for img in os.listdir(folder1) if img.endswith(('jpg', 'jpeg', 'png'))])
    images2 = sorted([img for img in os.listdir(folder2) if img.endswith(('jpg', 'jpeg', 'png'))])
    images3 = sorted([img for img in os.listdir(folder3) if img.endswith(('jpg', 'jpeg', 'png'))])
    images4 = sorted([img for img in os.listdir(folder4) if img.endswith(('jpg', 'jpeg', 'png'))])

    # 检查每个文件夹是否有相同数量的图片
    if len(images1) != len(images2) or len(images1) != len(images3) or len(images1) != len(images4):
        print("每个文件夹中的图片数量不相同！")
        return

    if not images1:
        print("没有找到图片文件！")
        return

    # 读取第一张图片来确定视频的尺寸
    first_image1 = cv2.imread(os.path.join(folder1, images1[0]))
    height, width, _ = first_image1.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height * 2))  # 视频尺寸为四个角拼接后的尺寸

    # 逐一拼接每个文件夹的图片并写入视频
    for i in range(len(images1)):
        # 读取每个文件夹中的图片
        image1 = cv2.imread(os.path.join(folder1, images1[i]))
        image2 = cv2.imread(os.path.join(folder2, images2[i]))
        image3 = cv2.imread(os.path.join(folder3, images3[i]))
        image4 = cv2.imread(os.path.join(folder4, images4[i]))

        # 创建一个大画布 (2x2拼接)
        canvas = 255 * np.ones(shape=(height * 2, width * 2, 3), dtype=np.uint8)  # 白色背景

        # 将四个文件夹中的图片放到画布的四个角落
        canvas[0:height, 0:width] = image1  # 左上角
        canvas[0:height, width:width*2] = image2  # 右上角
        canvas[height:height*2, 0:width] = image3  # 左下角
        canvas[height:height*2, width:width*2] = image4  # 右下角

        # 将拼接后的画面写入视频
        video_writer.write(canvas)

    # 释放资源
    video_writer.release()
    print(f"视频已成功创建：{output_video_path}")

def create_video_from_single_folder(folder, output_video_path, fps=30):
    # 检查文件夹路径是否存在
    if not os.path.exists(folder):
        print(f"文件夹 {folder} 不存在！")
        return

    # 获取文件夹中的图片文件，并按文件名排序
    def sort_key(filename):
        match = re.search(r'iter(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    images = sorted([img for img in os.listdir(folder) if img.endswith(('jpg', 'jpeg', 'png'))], key=sort_key)

    if not images:
        print("没有找到图片文件！")
        return

    # 读取第一张图片来确定视频的尺寸
    first_image = cv2.imread(os.path.join(folder, images[0]))
    height, width, _ = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # 视频尺寸为单张图片的尺寸

    # 逐一读取图片并写入视频
    for image_name in images:
        image = cv2.imread(os.path.join(folder, image_name))
        video_writer.write(image)

    # 释放资源
    video_writer.release()
    print(f"视频已成功创建：{output_video_path}")
# 示例使用
root = r'D:/code/CLIPasso/output_sketches/horse/horse_16strokes_seed0/'
folder1 = root + 'adj_logs'  # 替换为文件夹1的路径
folder2 = root + 'color_logs'  # 替换为文件夹2的路径
folder3 = root + 'matrix_pre_logs'  # 替换为文件夹3的路径
folder4 = root + 'matrix_after_logs'  # 替换为文件夹4的路径
folder5 = root + 'jpg_logs'  # 替换为文件夹4的路径
output_video_path = 'output_video.mp4'  # 输出视频路径
fps = 1  # 设置视频的帧率

# create_video_from_four_folders(folder1, folder2, folder3, folder4, output_video_path, fps)
# create_video_from_two_folders(folder1, folder2, output_video_path, fps)
create_video_from_single_folder(folder5, output_video_path, fps)
