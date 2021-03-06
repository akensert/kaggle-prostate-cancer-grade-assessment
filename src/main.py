import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn import model_selection, metrics
import sys

try:
    engine_type = sys.argv[1]
except:
    engine_type = 'efficientnet'

if engine_type == 'efficientnet':
    from models.efficientnet_sync import EfficientNetB4 as Engine
elif engine_type == 'inception':
    from models.inception_sync import InceptionV3 as Engine
elif engine_type == 'xception':
    from models.xception_sync import Xception as Engine
elif engine_type == 'inceptionresnet':
    from models.inceptionresnet_sync import InceptionResNetV2 as Engine
elif engine_type == 'densenet':
    from models.densenet_sync import DenseNet121 as Engine
elif engine_type == 'resnet':
    from models.resnet_sync import ResNet50 as Engine


from config import Config
from util import DataManager
from optimizer import get_optimizer
from generator import get_dataset
from model import NeuralNet, DistributedModel, Model

gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")

objective = Config.input.objective
input_shape = Config.input.input_shape
input_path = Config.input.path
random_state = Config.train.random_state
test_size = Config.train.test_size
fold = Config.train.fold
batch_size = Config.train.batch_size
epochs = Config.train.epochs

lr_max=Config.train.learning_rate.max
lr_min=Config.train.learning_rate.min
lr_decay_epochs=Config.train.learning_rate.decay_epochs
lr_warmup_epochs=Config.train.learning_rate.warmup_epochs
lr_power=Config.train.learning_rate.power

units=Config.model.units
dropout=Config.model.dropout
activation=Config.model.activation

train_data, valid_data = DataManager.get_train_data(
    split=True, test_size=test_size,
    random_state=random_state, add_image_size_info=True)

lr_steps_per_epoch=math.ceil(len(train_data) / Config.train.batch_size)

train_dataset = get_dataset(
    dataframe=train_data,
    input_path=input_path,
    batch_size=batch_size,
    training=True,
    augment='heavy',
    tta=1,
    input_size=input_shape,
    objective=objective,
    buffer_size=8192,
    cache=False,
)

valid_dataset = get_dataset(
    dataframe=valid_data,
    input_path=input_path,
    batch_size=batch_size,
    training=False,
    augment='light',
    tta=1,
    input_size=input_shape,
    objective=objective,
    buffer_size=1,
    cache=True,
)

with strategy.scope():

    optimizer = get_optimizer(
        steps_per_epoch=lr_steps_per_epoch,
        lr_max=lr_max,
        lr_min=lr_min,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=lr_warmup_epochs,
        power=lr_power
    )

    model = NeuralNet(
        engine=Engine,
        input_shape=input_shape,
        units=units,
        dropout=dropout,
        activation=activation,
        weights="imagenet")

    model.build([None, *input_shape])

    dist_model = DistributedModel(
        model, optimizer, strategy=strategy, mixed_precision=True,
        objective=Config.input.objective, label_smoothing=Config.input.label_smoothing)

    dist_model.fit_and_predict(
        fold, epochs, train_dataset, valid_dataset,
        data_provider=valid_data.data_provider,
        image_size=valid_data.image_size,
        )
