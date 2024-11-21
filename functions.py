#-*-coding: utf-8 -*-
import numpy as np
import gradio as gr
import cv2
from skimage import color
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils

def compare_images_delta_e(image_1, image_2):
    delta_e = color.deltaE_cie76(color.rgb2lab(image_1), color.rgb2lab(image_2))
    return delta_e.astype(np.uint8)

def generate_difference_heatmap(image_1, image_2):
    diff = np.abs(image_1.astype(np.float32) - image_2.astype(np.float32))
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
     
def Nearest_Neighbor_Interpolation(input_image, scale):
    H, W = input_image.shape[:2]
    new_W, new_H = int(W * scale), int(H * scale)
    output_image = np.zeros((new_H, new_W, 3), dtype=input_image.dtype)
    
    row_indices, col_indices = np.meshgrid(
        np.arange(new_H), np.arange(new_W), indexing='ij'
    )
    
    # 将新图的坐标映射到原图
    original_row_indices = (row_indices / scale).astype(int)
    original_col_indices = (col_indices / scale).astype(int)
    
    # 限制索引，防止越界
    original_row_indices = np.clip(original_row_indices, 0, H - 1)
    original_col_indices = np.clip(original_col_indices, 0, W - 1)
    
    # 使用高级索引从原图中获取对应的像素
    output_image = input_image[original_row_indices, original_col_indices]
    
    return output_image
    
def Bilinear_Resize(image, scale_factor):
    original_height, original_width = image.shape[:2]
    
    # 计算新图的尺寸
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    
    # 生成新图的网格坐标
    row_indices, col_indices = np.meshgrid(
        np.arange(new_height), np.arange(new_width), indexing='ij'
    )
    
    # 将新图的坐标映射到原图的浮点坐标
    original_row_coords = row_indices * (original_height / new_height)
    original_col_coords = col_indices * (original_width / new_width)
    
    # 计算浮点坐标的整数部分（左上角）和小数部分
    row_floor = np.floor(original_row_coords).astype(int)
    col_floor = np.floor(original_col_coords).astype(int)
    row_ceil = np.clip(row_floor + 1, 0, original_height - 1)
    col_ceil = np.clip(col_floor + 1, 0, original_width - 1)
    
    # 小数部分
    dy = original_row_coords - row_floor
    dx = original_col_coords - col_floor
    
    # 获取四个邻近像素的值
    top_left = image[row_floor, col_floor]
    top_right = image[row_floor, col_ceil]
    bottom_left = image[row_ceil, col_floor]
    bottom_right = image[row_ceil, col_ceil]
    
    # 进行双线性插值
    top = (1 - dx[:, :, None]) * top_left + dx[:, :, None] * top_right
    bottom = (1 - dx[:, :, None]) * bottom_left + dx[:, :, None] * bottom_right
    output_image = (1 - dy)[:, :, None] * top + dy[:, :, None] * bottom
    
    return output_image.astype(image.dtype)

def bicubic_weight(x):
    x = np.abs(x)
    w = np.zeros_like(x)
    mask1 = (x <= 1)
    mask2 = (x > 1) & (x < 2)
    
    w[mask1] = (1.5 * x[mask1] ** 3 - 2.5 * x[mask1] ** 2 + 1)
    w[mask2] = (-0.5 * x[mask2] ** 3 +2.5 * x[mask2] ** 2 - 4 * x[mask2] + 2)
    
    return w

def bicubic_resize(image, scale_factor):
    # 获取原图的尺寸
    original_height, original_width = image.shape[:2]
    
    # 计算新图的尺寸
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    
    # 生成新图的网格坐标
    row_indices, col_indices = np.meshgrid(
        np.arange(new_height), np.arange(new_width), indexing='ij'
    )
    
    # 将新图的坐标映射到原图的浮点坐标
    original_row_coords = row_indices * (original_height / new_height)
    original_col_coords = col_indices * (original_width / new_width)
    
    # 取整并计算偏移
    row_floor = np.floor(original_row_coords).astype(int)
    col_floor = np.floor(original_col_coords).astype(int)
    dy = original_row_coords - row_floor
    dx = original_col_coords - col_floor
    
    # 扩展 dy 和 dx 到 4x4 邻域
    dy_grid = dy[:, :, None] - np.array([-1, 0, 1, 2])[None, None, :]
    dx_grid = dx[:, :, None] - np.array([-1, 0, 1, 2])[None, None, :]
    
    # 计算行和列方向的权重
    weight_row = bicubic_weight(dy_grid)  # (new_height, new_width, 4)
    weight_col = bicubic_weight(dx_grid)  # (new_height, new_width, 4)
    
    # 计算最终的权重
    weights = weight_row[:, :, :, None] * weight_col[:, :, None, :]
    
    # 获取邻域的 4x4 像素网格坐标
    row_indices = np.clip(row_floor[:, :, None] + np.array([-1, 0, 1, 2])[None, None, :], 0, original_height - 1)
    col_indices = np.clip(col_floor[:, :, None] + np.array([-1, 0, 1, 2])[None, None, :], 0, original_width - 1)
    
    # 使用高级索引从原图中提取 4x4 邻域像素
    neighborhood = image[row_indices[:, :, :, None], col_indices[:, :, None, :]]
    
    weighted_sum = np.sum(weights[..., None] * neighborhood, axis=(2, 3))
    normalization = np.sum(weights, axis=(2, 3))
    
    # 防止除以零
    output_image = np.where(normalization[..., None] > 0, weighted_sum / normalization[..., None], 0)
    output_image = output_image.clip(0, 255).astype(np.uint8)
    return output_image

def lanczos_kernel(x, kernel_size=3):
    x = np.abs(x)
    result = np.where(x < kernel_size, np.sinc(x) * np.sinc(x / kernel_size), 0)
    return result

def lanczos_resize(image, scale_factor, kernel_size=3):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    row_indices, col_indices = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')
    
    origin_row_coord, origin_col_coord = row_indices / scale_factor, col_indices / scale_factor
    
    row_floor, col_floor = np.floor(origin_row_coord).astype(int), np.floor(origin_col_coord).astype(int)
    
    dx, dy = np.arange(-kernel_size + 1, kernel_size + 1), np.arange(-kernel_size + 1, kernel_size + 1) # 2 * kernel_size neighbor
    
    neighbor_row_coords = row_floor[:, :, None, None] + dx[None, None, :, None]
    neighbor_col_coords = col_floor[:, :, None, None] + dy[None, None, None, :]
    
    neighbor_row_coords = np.clip(neighbor_row_coords, 0, h - 1)
    neighbor_col_coords = np.clip(neighbor_col_coords, 0, w - 1)
    
    lanczos_row_weights = lanczos_kernel(origin_row_coord[:, :, None, None] - neighbor_row_coords, kernel_size)
    lanczos_col_weights = lanczos_kernel(origin_col_coord[:, :, None, None] - neighbor_col_coords, kernel_size)
    
    weights = lanczos_row_weights * lanczos_col_weights
    
    neighborhood = image[neighbor_row_coords, neighbor_col_coords]
    
    weighted_sum = np.sum(weights[:, :, :, :, None] * neighborhood, axis=(2, 3))
    normalization = np.sum(weights, axis=(2, 3))

    # 防止除以零
    output_image = np.where(normalization[:, :, None] > 0, weighted_sum / normalization[:, :, None], 0)
    output_image = output_image.clip(0, 255).astype(np.uint8)
    return output_image  


def function_hw2(image, implementation, method, scale_factor, kernel_size=None, rotation_angle=0, shear_factor=0, transform_type='none'):
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    image_resized = image
    
    if implementation == 'opencv':
        if method == 'NN':
            interpolation = cv2.INTER_NEAREST
        elif method == 'Bilinear':
            interpolation = cv2.INTER_LINEAR
        elif method == 'Bicubic':
            interpolation = cv2.INTER_CUBIC
        elif method == 'Lanczos':
            interpolation = cv2.INTER_LANCZOS4
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
    elif implementation == 'manual':
        if method == 'NN':
            image_resized = Nearest_Neighbor_Interpolation(image, scale_factor)
        elif method == 'Bilinear':
            image_resized = Bilinear_Resize(image, scale_factor)
        elif method == 'Bicubic':
            image_resized = bicubic_resize(image, scale_factor)
        elif method == 'Lanczos':
            image_resized = lanczos_resize(image, scale_factor, kernel_size)

    if transform_type == 'rotation':
        angle_rad = np.deg2rad(rotation_angle)
        center = (new_width / 2, new_height / 2)
        rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        transformed_image = cv2.warpAffine(image_resized, rot_mat, (new_width, new_height))
    
    elif transform_type == 'shear':
        shear_mat = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        transformed_image = cv2.warpAffine(image_resized, shear_mat, (new_width, new_height))
    else:
        transformed_image = image_resized

    return transformed_image

# generator code

class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution. nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
def generate_image_from_seed(seed, test_batch_size):
    # 加载训练好的GAN模型
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ngpu = 1
    netG_test = Generator(ngpu).to(device)
    netG_test.load_state_dict(torch.load('checkpoint/dcgan_checkpoint.pth', map_location=device))
    netG_test.eval()  # 切换到评估模式
    # generator.eval()
    torch.manual_seed(seed)
    noise = torch.randn(test_batch_size, 100, 1, 1, device=device)  # 假设输入是100维的噪声
    with torch.no_grad():
        fake = netG_test(noise).detach().cpu()

    # 将生成的图像制作成网格以便可视化
    vis = vutils.make_grid(fake, normalize=True)
    return ToPILImage()(vis)

def generate_editted_image(seed, test_batch_size, i=0, j=1):
    # 加载生成器模型
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ngpu = 1
    netG_test = Generator(ngpu).to(device)
    netG_test.load_state_dict(torch.load('/DATA/sqf/An-Image-Processing-Demo/checkpoint/dcgan_checkpoint.pth', map_location=device))
    netG_test.eval()
    torch.manual_seed(seed)
    # 随机生成64个噪声输入，生成对应的图像
    nz = 100
    noise = torch.randn(test_batch_size, nz, 1, 1, device=device)
    # 选择一对具有相反性质的图像 (例如 A+ 和 A-)
    noise_plus_A = noise[i].unsqueeze(0)  # 假设第 0 张图像具有属性 +A
    noise_minus_A = noise[j].unsqueeze(0)  # 假设第 1 张图像具有属性 -A

    # 计算编辑向量 vA
    vA = noise_plus_A.mean(0) - noise_minus_A.mean(0)
    edited_noise = noise + vA.unsqueeze(0)
    with torch.no_grad():
        edited_images = netG_test(edited_noise).detach().cpu()
    vis = vutils.make_grid(edited_images, normalize=True)
    return ToPILImage()(vis)
    

def function_hw3(seed, test_batch_size, is_editting=False, i=1, j=2):
    try:
        if is_editting == False:
            generated_image = generate_image_from_seed(seed, test_batch_size)
            return np.array(generated_image)
        else:
            generated_image = generate_editted_image(seed, test_batch_size, i-1, j-1)
            return np.array(generated_image)
    except Exception as e:
        raise gr.Error(f"生成图像时出错: {e}")

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