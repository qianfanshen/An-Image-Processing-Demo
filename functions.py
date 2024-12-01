# -*-coding: utf-8 -*-
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
                       [-1, 5, -1],
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
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
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
    denominator = np.sqrt(R ** 2 + G ** 2 + B ** 2 - R * G - R * B - G * B) + 1e-6
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
    tmp3 = 3.0 * L - (tmp1 + tmp2)
    tmp1 = np.clip(tmp1, 0, 1)
    tmp2 = np.clip(tmp2, 0, 1)
    tmp3 = np.clip(tmp3, 0, 1)

    R, G, B = np.zeros_like(tmp1, dtype=np.float32), np.zeros_like(tmp2, dtype=np.float32), np.zeros_like(tmp3,
                                                                                                          dtype=np.float32)
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
    w[mask2] = (-0.5 * x[mask2] ** 3 + 2.5 * x[mask2] ** 2 - 4 * x[mask2] + 2)

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

    dx, dy = np.arange(-kernel_size + 1, kernel_size + 1), np.arange(-kernel_size + 1,
                                                                     kernel_size + 1)  # 2 * kernel_size neighbor

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


def function_hw2(image, implementation, method, scale_factor, kernel_size=None, rotation_angle=0, shear_factor=0,
                 transform_type='none'):
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
    netG_test.load_state_dict(
        torch.load('/DATA/sqf/An-Image-Processing-Demo/checkpoint/dcgan_checkpoint.pth', map_location=device))
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
            generated_image = generate_editted_image(seed, test_batch_size, i - 1, j - 1)
            return np.array(generated_image)
    except Exception as e:
        raise gr.Error(f"生成图像时出错: {e}")


def add_random_noise(image, noise_type='gaussian', mean=0, sigma=25):
    if noise_type == 'gaussian':
        # Add Gaussian noise
        gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image = np.clip(image.astype(np.float32) + gaussian_noise, 0, 255)
    elif noise_type == 'salt_pepper':
        # Add Salt and Pepper noise
        noisy_image = image.copy()
        prob = 0.02
        num_salt = np.ceil(prob * image.size * 0.5)
        num_pepper = np.ceil(prob * image.size * 0.5)

        # Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 255

        # Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 0
    elif noise_type == 'None':
        noisy_image = image.copy()
    else:
        raise ValueError("Unsupported noise type. Use 'gaussian' or 'salt_pepper'")

    return np.uint8(noisy_image)


def bilateral_filter(image, d, sigma_s, sigma_c):
    # Ensure the kernel size is odd
    if d % 2 == 0:
        d += 1

    # Precompute Gaussian distance weights (space weights)
    offset = d // 2
    x, y = np.meshgrid(np.arange(-offset, offset + 1), np.arange(-offset, offset + 1))
    space_weight = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))
    space_weight = np.repeat(space_weight[:, :, np.newaxis], 3, axis=2)
    # Initialize the filtered image
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # Pad the input image to handle borders (considering RGB channels)
    padded_image = np.pad(image, ((offset, offset), (offset, offset), (0, 0)), mode='reflect')
    height, width, channels = image.shape

    for i in range(height):
        for j in range(width):
            # for c in range(channels):
            # Extract the region of interest for the specific channel
            roi = padded_image[i:i + d, j:j + d, :]

            # Compute intensity weights based on the difference in pixel values for each channel
            intensity_weight = np.exp(-((roi - image[i, j, :]) ** 2) / (2 * sigma_c ** 2))

            # Compute the bilateral filter response
            total_weight = space_weight * intensity_weight
            total_weight /= np.sum(total_weight, axis=(0, 1))
            filtered_pixel = np.sum(total_weight * roi, axis=(0, 1))

            # Assign the filtered pixel to the output
            filtered_image[i, j, :] = filtered_pixel

    return np.uint8(filtered_image)


def box_filter(image, kernel_size):
    height, width, channels = image.shape
    r = kernel_size // 2

    # 构建积分图
    integral_image = np.cumsum(np.cumsum(image, axis=0), axis=1)

    # 对每个通道分别处理
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # 定义窗口边界
            y1, y2 = max(i - r, 0), min(i + r + 1, height)
            x1, x2 = max(j - r, 0), min(j + r + 1, width)

            # 计算窗口内的像素总和
            sum_region = (
                    integral_image[y2 - 1, x2 - 1, :]
                    - (integral_image[y1 - 1, x2 - 1, :] if y1 > 0 else 0)
                    - (integral_image[y2 - 1, x1 - 1, :] if x1 > 0 else 0)
                    + (integral_image[y1 - 1, x1 - 1, :] if y1 > 0 and x1 > 0 else 0)
            )

            # 计算均值
            filtered_image[i, j, :] = sum_region / ((y2 - y1) * (x2 - x1))

    return filtered_image


def guided_filter(I, P, kernel_size, eps=0.01):
    I = I.astype(np.float32) / 255.
    P = P.astype(np.float32) / 255.

    mean_I = box_filter(I, kernel_size)
    mean_P = box_filter(P, kernel_size)
    corr_I = box_filter(I * I, kernel_size)
    corr_IP = box_filter(I * P, kernel_size)
    var_I = corr_I - mean_I * mean_I
    cov_IP = corr_IP - mean_P * mean_I
    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I
    mean_a = box_filter(a, kernel_size)
    mean_b = box_filter(b, kernel_size)
    q = mean_a * I + mean_b
    return (q * 255.0).astype(np.uint8)


def function_hw4(input_image, method='bilateral', d=5, sigma_c=10, sigma_s=10):
    if input_image is None:
        raise gr.Error('请上传一张图像', duration=5)
    if method == 'bilateral_denoise':
        output_image = bilateral_filter(input_image, d, sigma_s, sigma_c)
    if method == 'bilateral_sharpen':
        base_layer = bilateral_filter(input_image, d, sigma_s, sigma_c)
        detail = input_image - base_layer
        output_image = np.clip(input_image + detail, 0, 255).astype(np.uint8)
    if method == 'guided_filter_denoise':
        output_image = guided_filter(input_image, input_image, d)
    if method == 'guided_filter_sharpen':
        base_layer = guided_filter(input_image, input_image, d)
        detail = input_image - base_layer
        output_image = np.clip(input_image + detail, 0, 255).astype(np.uint8)
    return output_image


def calc_hists(img, num_bins=256):
    hists = np.zeros(num_bins, dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hists[img[i, j]] += 1
    return hists


def clip_hists(hists, threshold_ratio=10.0):
    hists = hists.astype(np.float32)
    hists_clip = hists.copy()
    threshold_value = np.mean(hists) * threshold_ratio
    hists_clip[hists_clip > threshold_value] = threshold_value
    avg_extra = np.sum(hists - hists_clip) / hists.shape[0]
    hists_clip += avg_extra
    return hists_clip


def he(img, bins=256):
    hists = calc_hists(img, bins)
    cdf_normalized = cal_cdf(hists)
    img_ = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_[i, j] = int(cdf_normalized[img[i, j]] * (bins - 1))
    return img_


def clhe(img, threshold=10.0, bins=256):
    hists = calc_hists(img)
    hists_clip = clip_hists(hists, threshold)
    cdf_normalized = cal_cdf(hists_clip)
    img_ = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_[i, j] = int(cdf_normalized[img[i, j]] * (bins - 1))
    return img_



def ahe(img, block_size=8, bins=256):
    H, W = img.shape
    print(H, W)
    tile_h = int(H / block_size)
    tile_w =int(W / block_size)
    print(tile_h, tile_w)
    img_ = np.zeros_like(img)
    map=np.zeros((block_size, block_size, bins), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            s_i , e_i = int(i * tile_h), int((i+1) * tile_h)
            s_j , e_j = int(j * tile_w), int((j+1) * tile_w)
            sub_img = img[s_i:e_i, s_j:e_j]
            hists = calc_hists(sub_img, bins)
            cdf_normalized = cal_cdf(hists)
            map[i, j] = cdf_normalized * (bins -1)

    for i in range(H):
        for j in range(W):
            index_i = int(i//tile_h)
            index_j = int(j//tile_w)
            if (i <= tile_h/2 and j <= tile_w/2) or (i<=tile_h/2 and j>=W-tile_w/2) or (i>=H-tile_h/2 and j<=tile_w/2) or (i>=H-tile_h/2 and j>=W-tile_w/2):
                img_[i, j] = int(map[index_i, index_j, img[i, j]])
            elif i<=tile_h/2 or i>=H-tile_h/2:

                l_j = index_j if (j - index_j*tile_w)>tile_w/2  else index_j-1
                r_j = l_j+1
                w_l = 1 - (j - l_j * tile_w - tile_w/2) / tile_w
                w_r = 1 - w_l
                print(l_j, r_j, i, j,w_l, w_r)
                print(index_i)
                img_[i, j] = int(w_l*map[index_i, l_j, img[i, j]] + w_r*map[index_i, r_j, img[i, j]])
            elif j<=tile_w/2 or j>=W-tile_w/2:
                u_i = index_i if (i - index_i*tile_h)>tile_h/2 else index_i - 1
                b_i = u_i + 1
                w_u = 1 - (i - u_i* tile_h - tile_h/2) / tile_h
                w_b = 1 - w_u
                img_[i, j] = int(w_u * map[u_i, index_j, img[i, j]] + w_b * map[b_i, index_j, img[i, j]])
            else:
                l_j = index_j if j >= (j - index_j*tile_w)>tile_w/2 else index_j - 1
                r_j =  l_j + 1
                w_l = 1 - (j - l_j * tile_w-tile_w/2) / tile_w
                w_r = 1 - w_l
                u_i = index_i if (i - index_i*tile_h)>tile_h/2  else index_i - 1
                b_i = u_i + 1
                w_u = 1 - (i - u_i * tile_h-tile_h/2) / tile_h
                w_b = 1 - w_u
                img_[i,j] = int(w_l * (w_u * map[u_i, l_j, img[i, j]] + w_b * map[b_i, l_j, img[i, j]]) + w_r * (w_u * map[u_i, r_j, img[i, j]]  + w_b * map[b_i, r_j, img[i, j]] ))

    return img_


def clahe(img, block_size=8, threshold=10.0, bins=256):
    H, W = img.shape
    print(H, W)
    tile_h = int(H / block_size)
    tile_w = int(W / block_size)
    print(tile_h, tile_w)
    img_ = np.zeros_like(img)
    map = np.zeros((block_size, block_size, bins), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            s_i, e_i = int(i * tile_h), int((i + 1) * tile_h)
            s_j, e_j = int(j * tile_w), int((j + 1) * tile_w)
            sub_img = img[s_i:e_i, s_j:e_j]
            hists = calc_hists(sub_img, bins)
            hists_clip = clip_hists(hists, threshold)
            cdf_normalized = cal_cdf(hists_clip)
            map[i, j] = cdf_normalized * (bins - 1)

    for i in range(H):
        for j in range(W):
            index_i = int(i // tile_h)
            index_j = int(j // tile_w)
            if (i <= tile_h / 2 and j <= tile_w / 2) or (i <= tile_h / 2 and j >= W - tile_w / 2) or (
                    i >= H - tile_h / 2 and j <= tile_w / 2) or (i >= H - tile_h / 2 and j >= W - tile_w / 2):
                img_[i, j] = int(map[index_i, index_j, img[i, j]])
            elif i <= tile_h / 2 or i >= H - tile_h / 2:

                l_j = index_j if (j - index_j * tile_w) > tile_w / 2 else index_j - 1
                r_j = l_j + 1
                w_l = 1 - (j - l_j * tile_w - tile_w / 2) / tile_w
                w_r = 1 - w_l
                print(l_j, r_j, i, j, w_l, w_r)
                print(index_i)
                img_[i, j] = int(w_l * map[index_i, l_j, img[i, j]] + w_r * map[index_i, r_j, img[i, j]])
            elif j <= tile_w / 2 or j >= W - tile_w / 2:
                u_i = index_i if (i - index_i * tile_h) > tile_h / 2 else index_i - 1
                b_i = u_i + 1
                w_u = 1 - (i - u_i * tile_h - tile_h / 2) / tile_h
                w_b = 1 - w_u
                img_[i, j] = int(w_u * map[u_i, index_j, img[i, j]] + w_b * map[b_i, index_j, img[i, j]])
            else:
                l_j = index_j if j >= (j - index_j * tile_w) > tile_w / 2 else index_j - 1
                r_j = l_j + 1
                w_l = 1 - (j - l_j * tile_w - tile_w / 2) / tile_w
                w_r = 1 - w_l
                u_i = index_i if (i - index_i * tile_h) > tile_h / 2 else index_i - 1
                b_i = u_i + 1
                w_u = 1 - (i - u_i * tile_h - tile_h / 2) / tile_h
                w_b = 1 - w_u
                img_[i, j] = int(w_l * (w_u * map[u_i, l_j, img[i, j]] + w_b * map[b_i, l_j, img[i, j]]) + w_r * (
                            w_u * map[u_i, r_j, img[i, j]] + w_b * map[b_i, r_j, img[i, j]]))

    return img_


def cal_cdf(hist):
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]
    return cdf_normalized


def hist_match_channel(target, reference):
    target_hist, _ = np.histogram(target.flatten(), bins=256, range=[0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), bins=256, range=[0, 256])

    target_cdf = cal_cdf(target_hist)
    reference_cdf = cal_cdf(reference_hist)

    # 创建一个映射表
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diff = np.abs(reference_cdf - target_cdf[i])
        mapping[i] = np.argmin(diff)

    matched_channel = mapping[target]

    return matched_channel


def hist_match(target, reference):
    target_b, target_g, target_r = cv2.split(target)
    reference_b, reference_g, reference_r = cv2.split(reference)

    # 对每个通道进行直方图匹配
    matched_b = hist_match_channel(target_b, reference_b)
    matched_g = hist_match_channel(target_g, reference_g)
    matched_r = hist_match_channel(target_r, reference_r)

    # 合并三个通道，得到最终的匹配图像
    matched_image = cv2.merge([matched_b, matched_g, matched_r])

    return matched_image


def function_hw5(input_image, reference, method='clahe', tile_size=8, clip_limit=10.0):
    if input_image is None:
        raise gr.Error('请上传一张图像', duration=5)
    if method in ['clahe', 'he', 'clhe', 'ahe']:
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        v_channel = hsv_image[:, :, 2]
        if method == 'clahe':
            v_channel_clahe = clahe(v_channel, tile_size, clip_limit)
            hsv_image[:, :, 2] = v_channel_clahe
        elif method == 'he':
            v_channel_he = he(v_channel)
            hsv_image[:, :, 2] = v_channel_he
        elif method == 'clhe':
            v_channel_he = clhe(v_channel, clip_limit)
            hsv_image[:, :, 2] = v_channel_he
        elif method == 'ahe':
            v_channel_he = ahe(v_channel, tile_size)
            hsv_image[:, :, 2] = v_channel_he
        hsv_image[:, :, 1] = (hsv_image[:, :, 1].astype(np.float32) * 1.2).clip(0, 255).astype(np.uint8)
        output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        # output_image = clahe(gray_image, tile_size, clip_limit)
        # hsl_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        #
        # # 分离出HSV的三个通道：色调(H)，饱和度(S)，亮度(L)
        # h, s, l = cv2.split(hsl_img)
        #
        # # 创建CLAHE对象
        # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        #
        # # 对亮度通道（L）进行CLAHE处理
        # l_clahe = clahe.apply(l)
        #
        # # 合并处理后的亮度通道与色调和饱和度通道
        # hsl_img_clahe = cv2.merge([h, s, l_clahe])
        #
        # # 将图像从HSV转换回BGR
        # output_image = cv2.cvtColor(hsl_img_clahe, cv2.COLOR_HSV2BGR)
    elif method == 'match':
        target_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        reference_image = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        matched_image = hist_match(target_image, reference_image)
        output_image = cv2.cvtColor(matched_image, cv2.COLOR_RGB2BGR)
    return output_image
