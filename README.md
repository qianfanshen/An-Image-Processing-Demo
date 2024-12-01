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
- **Generate Human Faces**: Utilize the DCGAN architecture to generate an arbitrary number of human faces. Set a random seed that is used to control the generation process.

- **Face Feature Editing**: Choose two generated faces that exhibit distinct features (e.g., gender, age, facial expressions, etc.) and use these examples to modify or edit the rest of the generated faces.

## HW4 Image Denoising and Sharpening Tools
- **Adding Noise**: Introduce Gaussian noise or salt-and-pepper noise to the images to simulate real-world degradation.
- **Denoising/Sharpening with Bilateral Filter**: Utilize the manual bilateral filter to reduce noise while preserving edges or to enhance image sharpness effectively.
- **Denoising/Sharpening with Box-Filter-Based Guided Filte**r: Implement a guided filter based on box filtering for both noise removal and detail enhancement, ensuring edge preservation.


## HW5 Histogram Matching and Low-Light Enhancement tools
- **Histogram Matching**: Apply the histogram matching algorithm to stylize your images by adjusting their histograms to match a reference style.
- **Low-Light Enhancement tools**: Enhance underexposed images using advanced techniques such as HE, CLHE, AHE, and CLAHE to improve visibility and contrast in low-light conditions.