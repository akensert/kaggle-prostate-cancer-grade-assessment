from util import DataManager


class Config:

    class input:
        path = DataManager.get_path() + 'train_images_jpeg_1/' # change this
        tiff_format = False
        tiff_level = 1
        resize_ratio = 1
        input_shape = (1536, 1536, 3)
        patch_size = 256
        sample_size = 36
        preprocess_mode = 'float'

    class model:
        units = [512, 5]
        dropout = [0.5, 0.0]
        activation = ['relu', None]

    class train:
        seed = 42
        fold = 0
        epochs = 50
        batch_size = 2
        accum_steps = 1

        class learning_rate:
            max = 1e-4
            min = 1e-5
            decay_epochs = 50
            warmup_epochs = 1
            power = 1

    class infer:
        tta = 1 # not utilized yet
        predict_every = 1 # not utilized yet
