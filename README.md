### Prostate-Cancer-Grade-Assessment Challenge

#### Requirements

TensorFlow-GPU:
* CUDA-enabled Graphics Card (Nvidia)
* Nvidia GPU drivers
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

If 2., make sure to rename the image filenames like this (Bash):
```
for f in prostate-cancer-grade-assessment/input/prostate-cancer-grade-assessment/train_images/*; do mv "$f" "${f%_1.jpeg}.jpeg" ; done
```
or change the image\_path in generator.py


#### Tests

Csv files and image folder should be put in input/prostate-cancer-grade-assessment/. Also make sure that config.py has the correct path to the image folder.

To test if data are available, run from terminal:

```
python util.py
```

And to test if the model can be run succesfully, run from terminal:

```
python model.py
```
It should run without errors




