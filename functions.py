#-*-coding: utf-8 -*-
import numpy as np
import gradio as gr
import cv2
from skimage import color
from skimage.metrics import structural_similarity as ssim

def compare_images_delta_e(image):
    opencv_image = function_hw1(image, 'opencv')
    manual_image = function_hw1(image, 'manual')
    delta_e = color.deltaE_cie76(color.rgb2lab(opencv_image), color.rgb2lab(manual_image))
    return delta_e.astype(np.uint8)


def generate_difference_heatmap(image):
    opencv_image = function_hw1(image, 'opencv')
    manual_image = function_hw1(image, 'manual')
    diff = np.abs(opencv_image.astype(np.float32) - manual_image.astype(np.float32))
    print("差异值最小值：", diff.min())
    print("差异值最大值：", diff.max())

    heatmap = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

def blur_image(image, kernel_size=5):
    """
    使用高斯滤波模糊图像。

    参数：
    - image: 输入图像
    - kernel_size: 高斯核的大小（必须为奇数）

    返回：
    - 模糊后的图像
    """
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def sharpen_image(image):
    """
    使用锐化核对图像进行锐化。

    参数：
    - image: 输入图像

    返回：
    - 锐化后的图像
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def emboss_image(image):
    """
    对图像应用浮雕滤镜。

    参数：
    - image: 输入图像

    返回：
    - 浮雕效果的图像
    """
    kernel = np.array([[ -2, -1,  0],
                       [ -1,  1,  1],
                       [  0,  1,  2]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    embossed_image = cv2.add(embossed_image, 128)
    return embossed_image

def function_hw1(input_image, method='opencv'):
    if input_image is None:
        raise gr.Error('请上传一张图像', duration=5)
    if method == 'opencv':
        image_HSL = cv2.cvtColor(input_image, cv2.COLOR_BGR2HLS)
        output_image = cv2.cvtColor(image_HSL, cv2.COLOR_HLS2BGR)
    if method == 'manual':
        H, S, L = rgb_to_hsl(input_image)
        output_image = hsl_to_rgb(H, S, L)
        # output_image = cv2.cvtColor(image_HSL, cv2.COLOR_HLS2RGB)
    return output_image

def rgb_to_hsl(image):
    image = image / 255.0
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    L = image.mean(axis=2)
    
    S = np.zeros_like(L)
    S = 1 - (3 * np.min(image, axis=2) / (R + G + B + 1e-6))
    
    H = np.zeros_like(L)
    denominator = np.sqrt(R**2 + G**2 + B**2 - R * G - R * B - G * B) + 1e-6
    cos_theta = (R - 0.5 * G - 0.5 * B) / denominator
    cos_theta = np.clip(cos_theta, -1, 1)
    H = np.arccos(cos_theta)
    H[B > G] = 2 * np.pi - H[B > G]
    H = H / np.pi * 180.0

    return H, S, L

def hsl_to_rgb(H, S, L):
    mask_1 = (H >= 0) & (H < 120.0)
    mask_2 = (H >= 120.0) & (H < 240.0)
    mask_3 = (H >= 240.0) & (H < 360.0)
    
    H_ = H % 120
    tmp1 = L * (1.0 + S * np.cos(H_ / 180.0 * np.pi) / np.cos(np.pi / 3.0 - H / 180.0 * np.pi))
    tmp2 = L * (1.0 - S)
    tmp3 = 3.0 * L - (tmp1 +tmp2)
    tmp1 = np.clip(tmp1, 0, 1)
    tmp2 = np.clip(tmp2, 0, 1)
    tmp3 = np.clip(tmp3, 0, 1)
    
    R, G, B = np.zeros_like(tmp1, dtype=np.float32), np.zeros_like(tmp2, dtype=np.float32), np.zeros_like(tmp3, dtype=np.float32)
    R[mask_1] = tmp1[mask_1]
    G[mask_1] = tmp3[mask_1]
    B[mask_1] = tmp2[mask_1]
    R[mask_2] = tmp2[mask_2]
    G[mask_2] = tmp1[mask_2]
    B[mask_2] = tmp3[mask_2]
    R[mask_3] = tmp3[mask_3]
    G[mask_3] = tmp2[mask_3]
    B[mask_3] = tmp1[mask_3]
    
    R, G, B = (R * 255).astype(np.uint8), (G * 255).astype(np.uint8), (B * 255).astype(np.uint8)
    
    return np.stack([R, G, B], axis=-1)
     
    

def function_hw2(input_image):
    if input_image is None:
        raise gr.Error('��������ڴ���֮ǰ��������һ��ͼ��', duration=5)    
    output_image = input_image
    # �벹����ҵ2��ͼ��������
    return output_image

def function_hw3(input_image):
    if input_image is None:
        raise gr.Error('��������ڴ���֮ǰ��������һ��ͼ��', duration=5)   
    output_image = input_image
    # �벹����ҵ3��ͼ��������
    return output_image

def function_hw4(input_image):
    if input_image is None:
        raise gr.Error('��������ڴ���֮ǰ��������һ��ͼ��', duration=5)
    output_image = input_image
    # �벹����ҵ4��ͼ��������
    return output_image

def function_hw5(input_image):
    if input_image is None:
        raise gr.Error('��������ڴ���֮ǰ��������һ��ͼ��', duration=5)
    output_image = input_image
    # �벹����ҵ5��ͼ��������
    return output_image