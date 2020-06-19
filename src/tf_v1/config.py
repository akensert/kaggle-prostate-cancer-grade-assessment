from util import DataManager
import os
import glob


class Config:

    class input:

        path = DataManager.get_path()
        # for convenience when uploaded to kaggle for submission/inference
        if os.path.exists(path + 'test_images/'):
            path += 'test_images/'
        else:
            path += 'train_images/'
        print("Path to images: ", path)

        # for convenience when uploaded to kaggle for submission/inference
        if glob.glob(path+'*.jpeg'):
            tiff_format = False
        else:
            tiff_format = True
        print("Image tiff-format: ", tiff_format)

        tiff_level = 1 # onlt if tiff_format is Ture
        resize_ratio = 1
        input_shape = (1536, 1536, 3)
        patch_size = 256
        sample_size = 36

        ## inceptionv3 = 'tf'
        ## resnet50 = 'caffe'
        ## densenet121 = 'torch'
        ## xception = 'tf'
        ## inceptionresnet = 'tf'
        ## efficientnet = 'torch' or 'float' or 'none'
        preprocess_mode = 'float'
        label_smoothing = 0.0
        objective = 'mse'

    class model:
        units = [1]
        dropout = [0.2]
        activation = [None]

    class train:
        random_state = 102
        fold = 0
        epochs = 30
        batch_size = 2

        class learning_rate:
            max = 1e-4
            min = 1e-5
            decay_epochs = 30
            warmup_epochs = 1
            power = 1

    class infer:
        tta = 1 # not utilized yet
        predict_every = 1 # not utilized yet
