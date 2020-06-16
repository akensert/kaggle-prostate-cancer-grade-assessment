import tensorflow as tf
import numpy as np
import tqdm
import os
from sklearn import metrics
from abc import ABCMeta, abstractmethod


class NeuralNet(tf.keras.Model):

    def __init__(self, engine, input_shape, units,
                 dropout, activation, weights=None):

        super(NeuralNet, self).__init__()

        self.engine = engine(
            include_top=False,
            input_shape=input_shape,
            weights=weights)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        self.head = tf.keras.Sequential()
        for drop, unit, actv in zip(dropout, units, activation):
            self.head.add(tf.keras.layers.Dropout(drop))
            self.head.add(tf.keras.layers.Dense(unit, actv, dtype='float32'))

    def call(self, inputs, **kwargs):
        x = self.engine(inputs)
        x = self.pool(x)
        return self.head(x)


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, keras_model, optimizer, strategy, mixed_precision):
        self.keras_model = keras_model
        if optimizer is not None:
            if mixed_precision:
                self.optimizer = \
                    tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                        optimizer, loss_scale='dynamic')
            else:
                self.optimizer = optimizer
        if strategy is not None:
            self.strategy = strategy
            self.num_replicas_in_sync = self.strategy.num_replicas_in_sync
        else:
            self.num_replicas_in_sync = 1
        self.mixed_precision = mixed_precision

        if not(os.path.isdir('output/weights')):
            os.makedirs('output/weights')


    def _train_step(self, inputs):
        if self.mixed_precision:
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = self.keras_model(images, training=True)
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=logits)
                loss = (
                    tf.reduce_mean(cross_entropy)
                    * (1.0 / self.num_replicas_in_sync)
                )
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            scaled_gradients = tape.gradient(
                scaled_loss, self.keras_model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            self.optimizer.apply_gradients(
                zip(gradients, self.keras_model.trainable_variables))
            return loss
        else:
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = self.keras_model(images, training=True)
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=logits)
                loss = (
                    tf.reduce_mean(cross_entropy)
                    * (1.0 / self.num_replicas_in_sync)
                )
            gradients = tape.gradient(
                loss, self.keras_model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.keras_model.trainable_variables))
            return loss

    def _predict_step(self, inputs):
        images, labels = inputs
        logits = self.keras_model(images, training=False)
        probs = tf.math.sigmoid(logits)
        return probs, labels

    def inference(self, test_ds, weights=None, return_probs=False):
        print("Inference will be done without tf.distribute.strategy")
        self.keras_model.load_weights(weights)
        self._predict_step = tf.function(self._predict_step)
        test_ds = tqdm.tqdm(test_ds)
        preds, trues = list(), list()
        for inputs in test_ds:
            probs, labels = self._predict_step(inputs)
            if return_probs:
                preds.extend(probs.numpy().tolist())
                trues.extend(labels.numpy().tolist())
            else:
                preds.extend(tf.reduce_sum(probs, -1).numpy().tolist())
                trues.extend(tf.reduce_sum(labels, -1).numpy().tolist())
        return preds, trues


class Model(BaseModel):

    def __init__(self, keras_model, optimizer=None, strategy=None, mixed_precision=False):
        super(Model, self).__init__(
            keras_model=keras_model,
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=mixed_precision)

    def fit_and_predict(self, fold, epochs, train_ds, test_ds):

        self._train_step = tf.function(self._train_step)
        self._predict_step = tf.function(self._predict_step)

        score = 0.
        best_score = 0.
        for epoch_num in range(epochs):
            train_ds = tqdm.tqdm(train_ds)
            test_ds = tqdm.tqdm(test_ds)
            epoch_loss = 0.
            for i, inputs in enumerate(train_ds):
                loss = self._train_step(inputs)
                epoch_loss += loss
                train_ds.set_description(
                  "Score {:.6f} : Loss {:.6f}".format(score, epoch_loss/(i+1)))

            preds, trues = list(), list()
            for inputs in test_ds:
                probs, labels = self._predict_step(inputs)
                preds.extend(tf.reduce_sum(probs, -1).numpy().tolist())
                trues.extend(tf.reduce_sum(labels, -1).numpy().tolist())

            preds = np.round(preds, 0)
            trues = np.round(trues, 0)
            score = metrics.cohen_kappa_score(trues, preds, weights='quadratic')

            if score > best_score:
                best_score = score
                self.keras_model.save_weights(f'output/weights/model-{fold}-{epoch_num}.h5')

            with open('output/scores.txt', 'a') as f:
                f.write(
                    str(fold)  + ','  +
                    str(epoch_num) + ','  +
                    str(score)     + '\n'
                )


class DistributedModel(BaseModel):

    def __init__(self, keras_model, optimizer=None, strategy=None, mixed_precision=False):
        super(DistributedModel, self).__init__(
            keras_model=keras_model,
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=mixed_precision)

    def fit_and_predict(self, fold, epochs, train_ds, test_ds):

        @tf.function
        def distributed_train_step(inputs):
            loss_per_replica = self.strategy.run(self._train_step, args=(inputs,))
            loss_per_example = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss_per_replica, axis=None)
            return loss_per_example

        @tf.function
        def distributed_predict_step(inputs):
            probs, labels = self.strategy.run(self._predict_step, args=(inputs,))
            probs = probs.values
            labels = labels.values
            return probs, labels

        score = 0.
        best_score = 0.
        for epoch_num in range(epochs):
            train_dist_ds = self.strategy.experimental_distribute_dataset(train_ds)
            test_dist_ds = self.strategy.experimental_distribute_dataset(test_ds)
            train_dist_ds = tqdm.tqdm(train_dist_ds)
            test_dist_ds = tqdm.tqdm(test_dist_ds)
            epoch_loss = 0.
            for i, inputs in enumerate(train_dist_ds):
                loss = distributed_train_step(inputs)
                epoch_loss += loss
                train_dist_ds.set_description(
                  "Score {:.6f} : Loss {:.6f}".format(score, epoch_loss/(i+1)))

            preds, trues = list(), list()
            for inputs in test_dist_ds:
                probs, labels = distributed_predict_step(inputs)
                for prob, label in zip(probs, labels):
                    preds.extend(tf.reduce_sum(prob, -1).numpy().tolist())
                    trues.extend(tf.reduce_sum(label, -1).numpy().tolist())

            preds = np.round(preds, 0)
            trues = np.round(trues, 0)
            score = metrics.cohen_kappa_score(trues, preds, weights='quadratic')

            if score > best_score:
                best_score = score
                self.keras_model.save_weights(f'output/weights/model-{fold}-{epoch_num}.h5')

            with open('output/scores.txt', 'a') as f:
                f.write(
                    str(fold)  + ','  +
                    str(epoch_num) + ','  +
                    str(score)     + '\n'
                )
