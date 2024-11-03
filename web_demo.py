from __future__ import annotations

import argparse
import pathlib
import gradio as gr
from UI import *
from functions import *
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="#f0f0f0",  # 这里设置为您希望的纯色，例如浅灰色
            body_background_fill_dark="#1a1a1a",  # 在暗模式下的背景颜色
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            body_text_color="#000000",  
            block_title_text_weight="600",
            block_title_text_color = "#000000",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )

seafoam = Seafoam()


HTML_DESCRIPTION = '''
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">
   图像处理网页演示工具
</h1>
<p>使用方式，在浏览器中打开 <a href="http://127.0.0.1:8080" target="_blank">http://127.0.0.1:8080</a> 即可</p>
<p>作者: 沈千帆</p>
<p>学号: 2200013220</p>
</div>
'''

MD_DESCRIPTION = '''
## 此网页演示提供以下图像处理工具:
- 作业1：色彩处理工具
- 作业2：几何变换工具
- 作业3：XXX工具
- 作业4：XXX工具
- 作业5：XXX工具
'''

def main():
    with gr.Blocks(theme=seafoam) as demo:
        gr.Markdown(HTML_DESCRIPTION) 
        gr.Markdown(MD_DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('作业1: 色彩处理工具'):
                create_demo_hw1(function_hw1)        
            with gr.TabItem('作业2: 几何变换工具'):
                create_demo_hw2(function_hw2)   
            with gr.TabItem('作业3: XXX工具'):
                create_demo_hw3(function_hw3)  
            with gr.TabItem('作业4: XXX工具'):
                create_demo_hw4(function_hw4) 
            with gr.TabItem('作业5: XXX工具'):
                create_demo_hw5(function_hw5)                                    

    demo.launch(server_port=8080)

if __name__ == '__main__':
    main()