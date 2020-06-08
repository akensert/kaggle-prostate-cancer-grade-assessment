from manager import DataManager


class Config:

    class model:
        layers: [5]
        dropout: [0.2]

    class input:
        path = DataManager.get_path() + 'train_images_jpeg_1/'
        tiff_format = False
        tiff_level = 1
        resize_ratio = 1
        input_shape = (1536, 1536, 3)
        patch_size = 256
        sample_size = 36
        preprocess_mode = 'float'

    class train:
        seed = 42
        folds = 5
        epochs = 50
        batch_size = 2
        accum_steps = 4

        class learning_rate:
            max = 3e-4
            min = 3e-5
            decay_epochs = 50
            warmup_epochs = 1
            power = 1

    class infer:
        tta = 1
        predict_every = 1
