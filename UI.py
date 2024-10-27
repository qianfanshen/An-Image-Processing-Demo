#-*-coding: utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np
from functions import function_hw1, compare_images_delta_e, generate_difference_heatmap, blur_image, sharpen_image, emboss_image

def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业一') 
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
                        return compare_images_delta_e(image)
                    elif mode == 'Heatmap':
                        return generate_difference_heatmap(image)
    
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


def create_demo_hw3(process):
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