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
or change the image\_path in generator.py


#### Tests

Csv files and image folder should be put in input/prostate-cancer-grade-assessment/. Also make sure that config.py has the correct path to the image folder, and that tiff\_format is set to True if images are in tiff format (Dataset 1.) or False if images are in jpeg format (Dataset 2.).

To test if data are available, run from terminal:

```
python util.py
```

And to test if the model can be run succesfully, run from terminal:

```
python model.py
```
It should run without errors




