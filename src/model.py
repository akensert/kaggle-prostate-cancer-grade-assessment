import tensorflow as tf
import numpy as np
import tqdm
from typing import Tuple, Union


class NeuralNet(tf.keras.Model):

    def __init__(self, engine, input_shape, units,
                 dropout, activation, weights=None):

        super(NeuralNet, self).__init__()

        self.engine = engine(
            include_top = False,
            input_shape=input_shape,
            weights=weights)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        self.head = tf.keras.Sequential()
        for drop, unit, actv in zip(dropout, units, activation):
            self.head.add(tf.keras.layers.Dropout(drop))
            self.head.add(tf.keras.layers.Dense(unit, actv, dtype='float32'))

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.engine(inputs)
        x = self.pool(x)
        return self.head(x)


def fit(model, dataset, loss_fn, optimizer, accum_steps):

    @tf.function
    def grad(model, loss_fn, x, y, accum_steps):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = loss_fn(y, pred)
            if accum_steps > 1:
                loss /= accum_steps
            scaled_loss = optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        return loss, gradients

    dataset = tqdm.tqdm(dataset)

    epoch_loss = 0
    current_lr = optimizer.learning_rate(optimizer.iterations).numpy()
    if accum_steps <= 1:
        for i, (x, y) in enumerate(dataset):
            # obtain loss and gradients
            loss_value, gradients = grad(model, loss_fn, x, y, accum_steps)

            # apply gradients immediately (no gradient accumulation)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += loss_value
            dataset.set_description(
                "GA {} : LR {:.7f} : LOSS {:.6f}".format("Off", current_lr, epoch_loss/(i+1)))
    else:
        for i, (x, y) in enumerate(dataset):
            if i % accum_steps == 0:
                # (re-)initialize gradient accumulator
                grad_accum = [
                    tf.Variable(tf.zeros_like(tv), trainable=False)
                    for tv in model.trainable_variables]

            # obtain loss and gradients
            loss_value, gradients = grad(model, loss_fn, x, y, accum_steps)

            # add gradients to accumulator
            grad_accum = [
                grad_accum[i].assign_add(g) for i, g in enumerate(gradients)]

            # every accum_steps we apply the gradients
            if (i+1) % accum_steps == 0:
                optimizer.apply_gradients(
                    [(ga, tv) for ga, tv in zip(grad_accum, model.trainable_variables)])

            epoch_loss += loss_value*accum_steps
            dataset.set_description(
                "GA {} : LR {:.7f} : LOSS {:.6f}".format("On", current_lr, epoch_loss/(i+1)))


def predict(model, dataset):

    @tf.function
    def predict_step(model, x):
        return model(x, training=False)

    dataset = tqdm.tqdm(dataset)

    # initialize accumulators
    preds = tf.zeros([0,], dtype=tf.dtypes.float32)
    trues = tf.zeros([0,], dtype=tf.dtypes.float32)

    for i, (x, y) in enumerate(dataset):
        pred = predict_step(model, x)
        preds = tf.concat((preds, tf.reduce_sum(pred, -1)), axis=0)
        trues = tf.concat((trues, tf.reduce_sum(y, -1)),    axis=0)

    return preds.numpy(), trues.numpy()


if __name__ == "__main__":

    from generator import get_dataset
    from util import DataManager, get_optimizer
    from config import Config

    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    print("Turning on Mixed-Precision")
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    model = NeuralNet(
            engine=tf.keras.applications.ResNet50,
            input_shape=(224, 224, 3),
            units=[512, 5],
            dropout=[0.3, 0.3],
            activation=['relu', 'sigmoid'],
            weights='imagenet')

    model.build([None, 224, 224, 3])

    print(model.summary())

    # dummy data
    x = tf.random.uniform(shape=(4, 224, 224, 3), minval=0, maxval=1)
    print(">> shape of random data = {}".format(x.numpy().shape))

    training = False
    print(">> training = False (dropout Off)")
    for i in range(20):
        if i == 10:
            training = True
            print(">> training = True (dropout On)")
        pred = model(x, training=training)
        pred = pred.numpy()
        print("pred_shape={} : pred_1={}".format(pred.shape, pred[0][0]))


    print("Running fit and predict, using 'get_dataset' and 'get_optimizer'")
    data = DataManager.get_train_file()
    data = data.iloc[:4]
    print(">> shape of data = {}".format(data.shape))
    dataset = get_dataset(
            dataframe=data,
            input_path=Config.input.path,
            batch_size=1,
            training=True,
            augment=True,
            buffer_size=1024,
            cache=False,
        )

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = get_optimizer(
        steps_per_epoch=2,
        lr_max=1e-4,
        lr_min=1e-5,
        decay_epochs=1,
        warmup_epochs=1,
        power=1
    )
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    model = NeuralNet(
            engine=tf.keras.applications.ResNet50,
            input_shape=Config.input.input_shape,
            units=[5],
            dropout=[0.2],
            activation=['sigmoid'],
            weights='imagenet')
    model.build([None, *Config.input.input_shape])
    print("Fit with accum steps")
    fit(model, dataset, loss_fn, optimizer, accum_steps=2)
    print("Fit without accum steps")
    fit(model, dataset, loss_fn, optimizer, accum_steps=1)
    print("Predict")
    preds, trues = predict(model, dataset)
    print("Predictions =", preds)
