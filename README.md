# An Image Processing Demo
This is my personal repository for homework of **PKU 2024 Fall Image Processing Course**. I create a web demo based on [gradio](https://gradio.app) which contains couples of image processing functions.


## Installation

```bash
git clone https://github.com/qianfanshen/An-Image-Processing-Demo.git
pip install -r requirements.txt
```

## Usage
```bash
python web_demo.py
```

## HW1 Color Processing Tools
- The manual implementation and OpenCV implementation of RGB and HSL space conversion.
- Blurring images
- Sharpening images
- Embossing images

## HW2 Geometry Processing Tools
- The manual implementation and OpenCV implementation of **Nearest-Neighbors** / **Bilinear** / **Bicubic** / **Lanczos** interpolation method to resize images
- Rotation of images
- Shearing of images

## HW3 Human Face Generation Using [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- Generate Human Faces: Utilize the DCGAN architecture to generate an arbitrary number of human faces. Set a random seed that is used to control the generation process.

- Face Feature Editing: Choose two generated faces that exhibit distinct features (e.g., gender, age, facial expressions, etc.) and use these examples to modify or edit the rest of the generated faces. 