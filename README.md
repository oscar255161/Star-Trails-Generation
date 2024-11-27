# ADIP Final Project: Star Trails Generation 

## Introduction  
This project focuses on transforming static starry sky images into dynamic star trail animations in GIF format using Python. The process involves three main stages:
1. **Foreground and Background Separation**: Detect and isolate the foreground using image processing techniques.
2. **Star Detection**: Identify the brightest stars in the image for animation.
3. **GIF/PNG Generation**: Create star trails with circular motion and save results in GIF and PNG formats.

The project was developed in Visual Studio 2022 and provides both automatic and manual modes for star selection, allowing users to customize the star trail animation.

---

## Features  
- **Automatic and Manual Modes**:
  - Automatically detect the brightest star as the center of rotation.
  - Allow users to manually select any star as the rotation center.
- **Dynamic Star Trails**:
  - Simulate star movement with circular paths.
  - Generate cumulative star trails in static PNG and animated GIF formats.
- **Optimized for Different Image Sizes**:
  - Dynamically adjust parameters like frame count and duration for input images of varying dimensions.

---

## Methodology  
### Architecture  
The workflow is divided into three main parts:

#### **Part 1: Foreground Mask Processing**
- **Sobel Filter**: Sharpens edges by detecting horizontal and vertical gradients.  
- **High-pass Filter (HPF)**: Retains high-frequency components to enhance edge details.  
- **Canny Edge Detection**: Locates precise foreground edges.  
- **Morphological Closing**: Smooths boundaries, removes noise, and connects fragmented regions.  
- **Connected Component Analysis**: Segments the image into distinct regions and generates a foreground mask.

#### **Part 2: Star Detection**
- **Grayscale Conversion and Binarization**: Enhances bright regions representing stars.  
- **Contour Detection**: Identifies potential stars in the binarized image.  
- **Contour Analysis**: Determines the brightest stars based on brightness and area.  

#### **Part 3: Star Trail Animation**
- **Parameter Adjustment**: Calculates parameters like frame count and rotation speed based on input image size.  
- **Trail Drawing**: Simulates star movement using circular paths and accumulates their trails over time.  
- **Output**: Saves cumulative trails as PNG and combines frames into a GIF animation.

---

## Results  

| Optimization Technique | Phase 1 Time | Phase 2 Time | Phase 3 Time |  
|-------------------------|--------------|--------------|--------------|  
| None                   | 6.5 s        | 132.5 s      | 1721 s       |  
| V1 (GIF Optimization)  | 6.68 s       | 131.37 s     | 14.39 s      |  
| V1 + V2 (Parallelism)  | 6.49 s       | 52.34 s      | 13.33 s      |  

### Key Results:  
1. **Optimized Performance**: Reduced processing time by applying parallelism and GIF optimization.  
2. **Robustness**: Successfully handled diverse scenarios, including auroras and varying brightness levels.

---

## User Interface  

### Features:  
- **Load Image**: Upload a starry sky image from the system.  
- **Auto Find Star**: Automatically detect the brightest star.  
- **Manual Mode**: Select any star as the center of rotation using the mouse.  



---

## Challenges  
1. **Foreground Segmentation**:  
   - Low contrast between foreground and background led to difficulties in edge detection.  
   - Over-detection of clouds as foreground objects in some cases.  
2. **Execution Time**:  
   - Reduced time significantly by optimizing animation generation and applying parallelism.

---

## Conclusion  
This project successfully demonstrates the use of image processing techniques for generating star trail animations. By balancing automation and customization, it achieves robust performance across diverse input images. Future improvements may include further refinement of segmentation methods and support for more complex visual effects.

