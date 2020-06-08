import tensorflow as tf
import numpy as np
import tqdm
from typing import Tuple, Union


class NeuralNet(tf.keras.Model):

    def __init__(self,
                 engine: tf.keras.Model,
                 input_shape: Tuple[int],
                 weights: Union[str, None] = None) -> None:

        super(NeuralNet, self).__init__()

        self.engine = engine(
            include_top = False,
            input_shape=input_shape,
            weights=weights
        )
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.drop = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid, dtype='float32')

    @tf.function
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.engine(inputs)
        x = self.pool(x)
        x = self.drop(x, training=kwargs.get('training', False))
        return self.out(x)


def fit(model: tf.keras.Model,
        dataset: tf.data.Dataset,
        loss_fn: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        accum_steps: int = 0) -> None:

    @tf.function
    def grad(model, loss_fn,
             x: tf.Tensor, y: tf.Tensor, accum_steps: int) -> tf.Tensor:
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
            loss_value, gradients = grad(model, loss_fn, x, y, accum_steps)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss_value
            dataset.set_description(
                "LR {:.7f} : LOSS {:.6f}".format(current_lr, epoch_loss/(i+1)))

    else:
        print(">> fitting with accumulated gradients")
        for i, (x, y) in enumerate(dataset):
            if i % accum_steps == 0:
                # (re-)initialize gradient accumulator
                grad_accum = [
                    tf.Variable(tf.zeros_like(tv), trainable=False)
                    for tv in model.trainable_variables
                ]
            # compute/obtain gradient (and loss)
            loss_value, gradients = grad(model, loss_fn, x, y, accum_steps)
            # add gradients to accumulator
            grad_accum = [
                grad_accum[i].assign_add(g) for i, g in enumerate(gradients)
            ]
            # every accum_steps we apply the gradients
            if (i+1) % accum_steps == 0:
                optimizer.apply_gradients(
                    [(ga, tv) for ga, tv in zip(grad_accum, model.trainable_variables)]
                )
            epoch_loss += loss_value*accum_steps
            dataset.set_description(
                "LR {:.7f} : LOSS {:.6f}".format(current_lr, epoch_loss/(i+1)))

def predict(model: tf.keras.Model,
            dataset: tf.data.Dataset) -> Tuple[np.ndarray]:

    @tf.function
    def predict_step(model, x: tf.Tensor) -> tf.Tensor:
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
