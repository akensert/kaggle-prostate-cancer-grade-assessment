import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn import model_selection, metrics

from models.efficientnet_norm import EfficientNetB0 as Engine

from config import Config
from util import DataManager
from optimizer import get_optimizer
from generator import get_dataset
from model import NeuralNet, SingleGPUModel, MultiGPUModel


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


input_shape = Config.input.input_shape
input_path = Config.input.path
seed = Config.train.seed
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

train_data, valid_data = DataManager.get_train_data(split=True)

lr_steps_per_epoch=math.ceil(len(train_data) / Config.train.batch_size)

train_dataset = get_dataset(
    dataframe=train_data,
    input_path=input_path,
    batch_size=batch_size,
    training=True,
    augment=True,
    tta=1,
    input_size=input_shape,
    buffer_size=8192,
    cache=False,
)

valid_dataset = get_dataset(
    dataframe=valid_data,
    input_path=input_path,
    batch_size=batch_size,
    training=False,
    augment=False,
    tta=1,
    input_size=input_shape,
    buffer_size=1,
    cache=True,
)

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():

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
    weights='imagenet')
model.build([None, *input_shape])

# multi_gpu_model = MultiGPUModel(model, optimizer, strategy=strategy)
# multi_gpu_model.fit_and_predict(fold, epochs, train_dataset, valid_dataset)

single_gpu_model = SingleGPUModel(model, optimizer, strategy=None)
single_gpu_model.fit_and_predict(fold, epochs, train_dataset, valid_dataset)
