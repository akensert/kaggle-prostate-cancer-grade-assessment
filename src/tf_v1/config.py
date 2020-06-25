from util import DataManager
import os
import glob


class Config:

    class input:

        path = DataManager.get_path()

        if os.path.exists(path + 'test_images/'):
            path += 'test_images/'
        else:
            path += 'train_images/'
        print("Path to images: ", path)

        if glob.glob(path+'*.jpeg'):
            tiff_format = False
        else:
            tiff_format = True
        print("Image tiff-format: ", tiff_format)

        tiff_level = 1                      # only if tiff_format is Ture
        resize_ratio = 1                    # only if tiff_format is True
        input_shape = (1536, 1536, 3)
        patch_size = 256
        sample_size = 36
        preprocess_mode = 'float'
        objective = 'bce'
        label_smoothing = 0.0               # only if objective is 'cce'

    class model:
        units = [5]    # output dim should be [5] for 'bce', [6] for 'cce', or [1] for 'mse'
        dropout = [0.2]
        activation = [None]

    class train:
        random_state = 102
        fold = 0
        epochs = 20
        batch_size = 2

        class learning_rate:
            max = 3e-4
            min = 3e-5
            decay_epochs = 20
            warmup_epochs = 1
            power = 1

    class infer:
        tta = 1 # not utilized yet
        predict_every = 1 # not utilized yet
