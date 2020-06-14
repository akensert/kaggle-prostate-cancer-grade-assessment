### Prostate-Cancer-Grade-Assessment Challenge

#### Requirements

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

Dataset:<br>
1. https://www.kaggle.com/c/prostate-cancer-grade-assessment/data<br>
or
2. https://www.kaggle.com/lopuhin/panda-2020-level-1-2<br>

If 2., make sure to rename image filenames like this (Bash):
```
for f in input/prostate-cancer-grade-assessment/train_images/*; do mv "$f" "${f%_1.jpeg}.jpeg" ; done
```
or change the image\_path in generator.py<br>

Csv files and image folder should be put in input/prostate-cancer-grade-assessment/.


#### Train and predict with model

To train and predict with one of the models (e.g. the tf_v1 model), from the terminal do:
```
python src/tf_v1/main.py
```
