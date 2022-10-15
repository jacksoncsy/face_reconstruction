# ibug.face_reconstruction
This repo has the inference codes for 3D face reconstruction based on [DECA](https://arxiv.org/pdf/2012.04012.pdf) with some pretrained weights. We currently support blendshape model [FLAME](https://flame.is.tue.mpg.de/index.html). 

**\[Updates\]**
 - `2022.10.11`: Support inference with AR linear model

## Prerequisites
* [Numpy](https://www.numpy.org/): `pip install numpy`
* [OpenCV](https://opencv.org/): `pip install opencv-python`
* [PyTorch](https://pytorch.org/): `pip install torch torchvision`
* [scikit-image](https://scikit-image.org/): `pip install scikit-image`
* [Scipy](https://www.scipy.org/): `pip install scipy`
### Modules needed for test script
* [ibug.face_detection](https://github.com/hhj1897/face_detection): See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).
* [ibug.face_alignment](https://github.com/hhj1897/face_alignment): See this repository for details: [https://github.com/hhj1897/face_alignment](https://github.com/hhj1897/face_alignment).

## How to Install

1. Install the code
```bash
git clone https://github.com/jacksoncsy/face_reconstruction.git
cd face_reconstruction
pip install -e .
```

2. Download the 3DMMs assets \
    (a) Download [FLAME](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, move **generic_model.pkl** into ./ibug/face_reconstruction/deca/assets/flame  \
    (b) Download [FLAME landmark asset](https://github.com/YadiraF/DECA/blob/master/data/landmark_embedding.npy), and move it into ./ibug/face_reconstruction/deca/assets/flame \
    (c) (Optional) Download AR multilinear models and assets, and move them into ./ibug/face_reconstruction/deca/assets/ar_multilinear  
    (d) (Optional) Download AR linear models and assets, and move them into ./ibug/face_reconstruction/deca/assets/ar_linear

3. Download the pretrained models \
    (a) Download the [FLAME-based pretrained models](https://drive.google.com/drive/folders/1Gke3AwvtHvukz4XxGC4PwwgpFALR_xUL?usp=sharing), and put them into ./ibug/face_reconstruction/deca/weights  \
    (b) (Optional) Download the AR-based pretrained models, and put them into ./ibug/face_reconstruction/deca/weights

## How to Test
* To test on live video: `python face_reconstruction_test.py [-i webcam_index]`
* To test on a video file: `python face_reconstruction_test.py [-i input_file] [-o output_file]`

## How to Use

### Call 3D face reconstruction given 68 landmarks as input in Python file
```python
from ibug.face_reconstruction import DecaCoarsePredictor

# Instantiate the 3D reconstructor
reconstructor = DecaCoarsePredictor(device='cuda:0')

# Fit 3DMM to the face specified by the 68 2D landmarks.
image = cv2.imread("test.jpg")
reconstruction_results = reconstructor(image, landmarks, rgb=False)
```

## References
```
@inproceedings{DECA:Siggraph2021,
  title={Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author={Feng, Yao and Feng, Haiwen and Black, Michael J. and Bolkart, Timo},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)}, 
  volume = {40}, 
  number = {8}, 
  year = {2021}, 
  url = {https://doi.org/10.1145/3450626.3459936} 
}