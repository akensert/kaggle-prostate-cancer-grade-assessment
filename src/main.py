import numpy as np
import pandas as pd
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
import math
from sklearn import model_selection, metrics

from config import Config
from util import get_optimizer, DataManager
from generator import get_dataset
from model import NeuralNet, fit, predict


#tf.config.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


data = DataManager.get_train_file()

input_shape = Config.input.input_shape
input_path = Config.input.path
seed = Config.train.seed
folds = Config.train.folds
batch_size = Config.train.batch_size
epochs = Config.train.epochs
accum_steps = Config.train.accum_steps

lr_steps_per_epoch=math.ceil((10616 * (1-1/Config.train.folds)) / Config.train.batch_size) / accum_steps
lr_max=Config.train.learning_rate.max
lr_min=Config.train.learning_rate.min
lr_decay_epochs=Config.train.learning_rate.decay_epochs
lr_warmup_epochs=Config.train.learning_rate.warmup_epochs
lr_power=Config.train.learning_rate.power

units=Config.model.units
dropout=Config.model.dropout
activation=Config.model.activation

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = get_optimizer(
    steps_per_epoch=lr_steps_per_epoch,
    lr_max=lr_max,
    lr_min=lr_min,
    decay_epochs=lr_decay_epochs,
    warmup_epochs=lr_warmup_epochs,
    power=lr_power
)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
    optimizer, loss_scale='dynamic')

classes = np.where(data.data_provider == 'karolinska', 6, 0) + data.isup_grade.values
skf = model_selection.StratifiedKFold(
    folds, shuffle=True, random_state=seed).split(data.image_id, y=data.isup_grade)

for fold_num, (train_idx, valid_idx) in enumerate(skf):

    if fold_num == 0:
        model = NeuralNet(
            engine=EfficientNetB0,
            input_shape=input_shape,
            units=units,
            dropout=dropout,
            activation=activation,
            weights='noisy-student')
        model.build([None, *input_shape])

        train_dataset = get_dataset(
            dataframe=data.iloc[train_idx],
            input_path=input_path,
            batch_size=batch_size,
            training=True,
            augment=True,
            buffer_size=1024,
            cache=False,
        )

        valid_dataset = get_dataset(
            dataframe=data.iloc[valid_idx],
            input_path=input_path,
            batch_size=batch_size,
            training=False,
            augment=False,
            buffer_size=1,
            cache=True,
        )

        best_score = float('-inf')
        for epoch_num in range(epochs):

            fit(model, train_dataset, loss_fn, optimizer, accum_steps)

            preds, trues = predict(model, valid_dataset)

            if Config.infer.tta > 1:
                preds = preds.reshape((-1, Config.infer.tta)).mean(-1)
                trues = trues.reshape((-1, Config.infer.tta)).mean(-1)

            preds = np.round(preds, 0)
            trues = np.round(trues, 0)
            score = metrics.cohen_kappa_score(trues, preds, weights='quadratic')

            fold_num  = str(fold_num)  + ','
            epoch_num = str(epoch_num) + ','
            score     = str(score)     + '\n'
            with open('output/scores.txt', 'a') as f:
                f.write(fold+epoch+score)

            if score > best_score:
                best_score = score
                model.save_weights(f'output/weights/model-{fold_num}-{epoch_num}.h5')
