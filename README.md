## Prostate-Cancer-Grade-Assessment Challenge

### Requirements

Python >= 3.6<br>

TensorFlow-GPU:
* CUDA-enabled Graphics Card (Nvidia)
* Nvidia GPU drivers (>=418 if CUDA 10.1 or >=440 if CUDA 10.2)
* CUDA toolkit
* cuDNN

see https://www.tensorflow.org/install/gpu for more information

OpenCV-Python and third-party Python packages (Linux):
```
apt install -y libsm6 libxext6 libxrender-dev # install dependencies for opencv-python
pip install -r requirements.txt # install python packages
```

### Dataset

Download either of the following datasets (preferably (B)):<br>
1. (A) https://www.kaggle.com/c/prostate-cancer-grade-assessment/data<br>
or
1. (B) https://www.kaggle.com/lopuhin/panda-2020-level-1-2<br>
2. Unzip files and put both the csv-files as well as the image folder inside `input/prostate-cancer-grade-assessment/`
3. If (B), make sure to rename image filenames like this (Bash): `for f in input/prostate-cancer-grade-assessment/train_images/*; do mv "$f" "${f%_1.jpeg}.jpeg" ; done`<br>

### Modeling
1. Navigate to `src/tf_v1/`
2. [Optional; Dataset (B) required] Clean up pen marks in images with pen marks. Navigate into `scripts/` and run `python remove_penmarks.py`.
3. Finally, start the modeling (fit and predict) by running `python main.py` from `src/tf_v1/`.
