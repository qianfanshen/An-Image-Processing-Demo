#-*-coding: utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np
from functions import *


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业1') 
        gr.Markdown('### Part1. RGB到HLS颜色空间转换工具(手工实现与OPENCV库函数的对比)')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
        with gr.Row():
            with gr.Column():
                output_image_opencv = gr.Image(type='numpy', label='OpenCV 输出图像', interactive=False)
                run_button_opencv = gr.Button(value='OpenCV 处理', variant="primary")
                run_button_opencv.click(fn=function_hw1,
                                        inputs=[input_image, gr.State(value='opencv')],
                                        outputs=[output_image_opencv])
            with gr.Column():
                output_image_manual = gr.Image(type='numpy', label='手工实现 输出图像', interactive=False)
                run_button_manual = gr.Button(value='手工实现 处理', variant="primary")
                run_button_manual.click(fn=function_hw1,
                                        inputs=[input_image, gr.State(value='manual')],
                                        outputs=[output_image_manual])
                
            with gr.Column():
                diff_image = gr.Image(type='numpy', label='差异图像 (OpenCV 与 手工实现)', interactive=False)
                compare_mode = gr.Dropdown(choices=['减法', 'Delta E', 'Heatmap'], label='选择对比方式')
                compare_button = gr.Button(value='对比处理', variant="primary")
    
                def compare_images(image, mode):
                    if mode == '减法':
                        return np.abs(function_hw1(image, 'opencv') - function_hw1(image, 'manual'))
                    elif mode == 'Delta E':
                        return compare_images_delta_e(function_hw1(image, 'opencv'),function_hw1(image, 'manual'))
                    elif mode == 'Heatmap':
                        return generate_difference_heatmap(function_hw1(image, 'opencv'),function_hw1(image, 'manual'))
    
                compare_button.click(fn=compare_images, inputs=[input_image, compare_mode], outputs=[diff_image])
        
        gr.Markdown('### Part2. 简单的图像处理功能（使用OPENCV库）')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
        with gr.Row():
            with gr.Column():
                output_image_blur = gr.Image(type='numpy', label='模糊（Blurring）', interactive=False)
                kernel = gr.Slider(minimum=1, maximum=99, step=2, value=1, label="选择一个奇数作为高斯核的大小")
                run_button = gr.Button(value='Process', variant="primary")
                run_button.click(fn=blur_image,
                                        inputs=[input_image, kernel],
                                        outputs=[output_image_blur])
            with gr.Column():
                output_image_sharpen = gr.Image(type='numpy', label='锐化（Sharpening）', interactive=False)
                run_button = gr.Button(value='Process', variant="primary")
                run_button.click(fn=sharpen_image,
                                        inputs=[input_image],
                                        outputs=[output_image_sharpen])
                
            with gr.Column():
                output_image_sharpen = gr.Image(type='numpy', label='浮雕（emboss）', interactive=False)
                run_button = gr.Button(value='Process', variant="primary")
                run_button.click(fn=emboss_image, inputs=[input_image], outputs=[output_image_sharpen])
        


    return demo


def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业2')
        gr.Markdown('### Part1. 使用插值算法实现图像放缩功能')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
                scale_factor = gr.Slider(minimum=0.5, maximum=4.0, step=0.1, label='缩放因子', value=1.0)
        gr.Markdown('- **使用OPENCV函数**')
        with gr.Row():
            with gr.Column():
                output_image_nn_opencv = gr.Image(type='numpy', label='最近邻插值 (Nearest Neighbor)', interactive=False)
                run_button_nn_opencv = gr.Button(value='Process', variant="primary")
                method = gr.State(value='NN')
                run_button_nn_opencv.click(
                    fn=function_hw2,
                    inputs=[input_image, gr.State(value='opencv'), method, scale_factor],
                    outputs=[output_image_nn_opencv])
            with gr.Column():
                output_image_bilinear_opencv = gr.Image(type='numpy', label='双线性插值 (Bilinear)', interactive=False)
                run_button_bilinear_opencv = gr.Button(value='Process', variant="primary")
                method = gr.State(value='Bilinear')
                run_button_bilinear_opencv.click(
                    fn=function_hw2,
                    inputs=[input_image, gr.State(value='opencv'), method, scale_factor],
                    outputs=[output_image_bilinear_opencv])
            with gr.Column():
                output_image_bicubic_opencv = gr.Image(type='numpy', label='双三次插值 (Bicubic)', interactive=False)
                run_button_bicubic_opencv = gr.Button(value='Process', variant="primary")
                method = gr.State(value='Bicubic')
                run_button_bicubic_opencv.click(
                    fn=function_hw2,
                    inputs=[input_image, gr.State(value='opencv'), method, scale_factor],
                    outputs=[output_image_bicubic_opencv])
                
        gr.Markdown('- **使用手工实现函数**')
        with gr.Row():
            with gr.Column():
                output_image_nn_manual = gr.Image(type='numpy', label='最近邻插值 (Nearest Neighbor)', interactive=False)
                run_button_nn_manual = gr.Button(value='Process', variant="primary")
                method = gr.State(value='NN')
                run_button_nn_manual.click(
                    fn=function_hw2,
                    inputs=[input_image, gr.State(value='manual'), method, scale_factor],
                    outputs=[output_image_nn_manual])
            with gr.Column():
                output_image_bilinear_manual = gr.Image(type='numpy', label='双线性插值 (Bilinear)', interactive=False)
                run_button_bilinear_opencv = gr.Button(value='Process', variant="primary")
                method = gr.State(value='Bilinear')
                run_button_bilinear_opencv.click(
                    fn=function_hw2,
                    inputs=[input_image, gr.State(value='manual'), method, scale_factor],
                    outputs=[output_image_bilinear_manual])
            with gr.Column():
                output_image_bicubic_manual = gr.Image(type='numpy', label='双三次插值 (Bicubic)', interactive=False)
                run_button_bicubic_opencv = gr.Button(value='Process', variant="primary")
                method = gr.State(value='Bicubic')
                run_button_bicubic_opencv.click(
                    fn=function_hw2,
                    inputs=[input_image, gr.State(value='manual'), method, scale_factor],
                    outputs=[output_image_bicubic_manual])
                
        gr.Markdown('- **Lanczos插值**')
        with gr.Row():
            output_image_lanczos_opencv_state = gr.State()
            output_image_lanczos_manual_state = gr.State()
            with gr.Column():
                output_image_lanczos_opencv = gr.Image(type='numpy', label='OPENCV实现', interactive=False)
                run_button_lanczos_opencv = gr.Button(value='Process', variant="primary")
                method = gr.State(value='Lanczos')
                run_button_lanczos_opencv.click(
                    fn=lambda *args: [function_hw2(*args)] * 2,
                    inputs=[input_image, gr.State(value='opencv'), method, scale_factor],
                    outputs=[output_image_lanczos_opencv, output_image_lanczos_opencv_state])
            with gr.Column():
                output_image_lanczos_manual = gr.Image(type='numpy', label='手工实现', interactive=False)
                run_button_lanczos_manual = gr.Button(value='Process', variant="primary")
                kernel_size_lanczos = gr.Slider(minimum=2, maximum=4, step=1, value=3, label="选择2-4中的一个整数")
                method = gr.State(value='Lanczos')
                run_button_lanczos_manual.click(
                    fn=lambda *args: [function_hw2(*args)] * 2,
                    inputs=[input_image, gr.State(value='manual'), method, scale_factor, kernel_size_lanczos],
                    outputs=[output_image_lanczos_manual, output_image_lanczos_manual_state])
            with gr.Column():
                diff_image_lanczos = gr.Image(type='numpy', label='差异图像 (OpenCV 与 手工实现)', interactive=False)
                compare_mode = gr.Dropdown(choices=['减法', 'Delta E', 'Heatmap'], label='选择对比方式')
                compare_button_lanczos = gr.Button(value='Process', variant="primary")
    
                def compare_images(mode, image_opencv, image_manual):
                    if mode == '减法':
                        return np.abs(image_opencv - image_manual)
                    elif mode == 'Delta E':
                        return compare_images_delta_e(image_opencv, image_manual)
                    elif mode == 'Heatmap':
                        return generate_difference_heatmap(image_opencv, image_manual)

                    # 将存储的状态图像作为输入，避免重新计算
                compare_button_lanczos.click(
                    fn=compare_images,
                    inputs=[compare_mode, output_image_lanczos_opencv_state, output_image_lanczos_manual_state],
                    outputs=[diff_image_lanczos]
                )
            
        gr.Markdown('### Part2. 图像几何变换')
        
        # 几何变换选项
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
                transform_type = gr.Dropdown(choices=['none', 'rotation', 'shear'], label='选择变换类型')
                rotation_angle = gr.Slider(minimum=-180, maximum=180, step=1, label='旋转角度', value=0)
                shear_factor = gr.Slider(minimum=-1.0, maximum=1.0, step=0.1, label='斜切因子', value=0.0)

        # 输出
        output_image_transformed = gr.Image(type='numpy', label='变换后的图像', interactive=False)
        process_button = gr.Button("应用变换", variant="primary")
        
        # 点击事件
        process_button.click(
            fn=function_hw2,
            inputs=[input_image, gr.State(value='none'), gr.State(value='Bilinear'), scale_factor, gr.State(value=3), rotation_angle, shear_factor, transform_type],
            outputs=[output_image_transformed]
        )
    return demo


def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵ��: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=function_hw2,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵ��: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵ��: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo