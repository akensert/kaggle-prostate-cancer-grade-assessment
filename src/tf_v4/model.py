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
    def __init__(self, keras_model, optimizer, strategy,
                 mixed_precision, objective, label_smoothing):
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

        if objective == 'mse':
            self.loss_fn = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE)
        elif objective == 'bce':
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)
        elif objective == 'cce':
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)
        else:
            raise ValueError("objective has to be either 'mse', 'bce' or 'cce'")

        self.objective = objective
        self.label_smoothing = label_smoothing

        if not(os.path.isdir('output/weights')):
            os.makedirs('output/weights')


    def _train_step(self, inputs):

        images, labels = inputs

        if self.label_smoothing > 0.0 and self.objective == 'cce':
            labels = (
                labels * (1 - self.label_smoothing)
                + 0.5 * self.label_smoothing
            )

        if self.mixed_precision:
            with tf.GradientTape() as tape:
                logits = self.keras_model(images, training=True)
                loss = (
                    tf.reduce_mean(self.loss_fn(labels, logits))
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
            with tf.GradientTape() as tape:
                logits = self.keras_model(images, training=True)
                loss = (
                    tf.reduce_mean(self.loss_fn(labels, logits))
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
        if self.objective == 'bce':
            return tf.math.sigmoid(logits), labels
        elif self.objective == 'cce':
            return tf.nn.softmax(logits), labels
        return logits, labels

    def inference(self, test_ds, weights=None, return_probs=False):
        print("Inference will be done without tf.distribute.strategy")
        self.keras_model.load_weights(weights)
        self._predict_step = tf.function(self._predict_step)
        test_ds = tqdm.tqdm(test_ds)
        preds_list, trues_list = list(), list()
        for inputs in test_ds:
            preds, labels = self._predict_step(inputs)
            if return_probs and (self.objective == 'bce' or self.objective == 'cce'):
                preds_list.extend(preds.numpy().tolist())
                trues_list.extend(labels.numpy().tolist())
            elif self.objective == 'cce':
                preds_list.extend(tf.math.argmax(preds, -1).numpy().tolist())
                trues_list.extend(tf.math.argmax(labels, -1).numpy().tolist())
            else:
                preds_list.extend(tf.reduce_sum(preds, -1).numpy().tolist())
                trues_list.extend(tf.reduce_sum(labels, -1).numpy().tolist())
        return preds_list, trues_list


class Model(BaseModel):

    def __init__(self, keras_model, optimizer=None, strategy=None,
                 mixed_precision=False, objective='bce', label_smoothing=0.0):
        super(Model, self).__init__(
            keras_model=keras_model,
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=mixed_precision,
            objective=objective,
            label_smoothing=label_smoothing)

    def fit_and_predict(self, fold, epochs, train_ds, test_ds):

        self._train_step = tf.function(self._train_step)
        self._predict_step = tf.function(self._predict_step)

        score = 0.
        best_score = 0.
        for epoch in range(epochs):
            train_ds = tqdm.tqdm(train_ds)
            test_ds = tqdm.tqdm(test_ds)
            epoch_loss = 0.
            for i, inputs in enumerate(train_ds):
                loss = self._train_step(inputs)
                epoch_loss += loss
                train_ds.set_description(
                  "Score {:.6f} : Loss {:.6f}".format(score, epoch_loss/(i+1)))

            preds_list, trues_list = list(), list()
            for inputs in test_ds:
                preds, labels = self._predict_step(inputs)
                if self.objective == 'cce':
                    preds_list.extend(tf.math.argmax(preds, -1).numpy().tolist())
                    trues_list.extend(tf.math.argmax(labels, -1).numpy().tolist())
                else:
                    preds_list.extend(tf.reduce_sum(preds, -1).numpy().tolist())
                    trues_list.extend(tf.reduce_sum(labels, -1).numpy().tolist())

            if self.objective != 'cce':
                preds_list = np.clip(np.round(preds_list, 0), 0, 5)
                trues_list = np.clip(np.round(trues_list, 0), 0, 5)

            score = metrics.cohen_kappa_score(
                trues_list, preds_list, weights='quadratic')

            if score > best_score:
                best_score = score
                self.keras_model.save_weights(f'output/weights/model-{fold}-{epoch}.h5')

            with open('output/scores.txt', 'a') as f:
                f.write(
                    str(fold)      + ','  +
                    str(epoch) + ','  +
                    str(score)     + '\n'
                )


class DistributedModel(BaseModel):

    def __init__(self, keras_model, optimizer=None, strategy=None,
                 mixed_precision=False, objective='bce', label_smoothing=0.0):
        super(DistributedModel, self).__init__(
            keras_model=keras_model,
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=mixed_precision,
            objective=objective,
            label_smoothing=label_smoothing)

    def fit_and_predict(self, fold, epochs, train_ds, test_ds,
                        data_provider=None, image_size=None):

        @tf.function
        def distributed_train_step(inputs):
            loss_per_replica = self.strategy.run(self._train_step, args=(inputs,))
            loss_per_example = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss_per_replica, axis=None)
            return loss_per_example

        @tf.function
        def distributed_predict_step(inputs):
            preds, labels = self.strategy.run(self._predict_step, args=(inputs,))
            if tf.is_tensor(preds):
                return [preds], [labels]
            else:
                return preds.values, labels.values

        score = 0.
        best_score = 0.
        for epoch in range(epochs):
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

            preds_list, trues_list = list(), list()
            for inputs in test_dist_ds:
                preds, labels = distributed_predict_step(inputs)
                for pred, label in zip(preds, labels):
                    if self.objective == 'cce':
                        preds_list.extend(tf.math.argmax(pred, -1).numpy().tolist())
                        trues_list.extend(tf.math.argmax(label, -1).numpy().tolist())
                    else:
                        preds_list.extend(tf.reduce_sum(pred, -1).numpy().tolist())
                        trues_list.extend(tf.reduce_sum(label, -1).numpy().tolist())

            if self.objective != 'cce':
                preds_list = np.clip(np.round(preds_list, 0), 0, 5)
                trues_list = np.clip(np.round(trues_list, 0), 0, 5)

            if image_size is not None:
                small_idx = np.where(
                     image_size <= 0.75)[0]
                medium_idx = np.where(
                    (image_size >  0.75) & (image_size <= 1.25))[0]
                large_idx = np.where(
                     image_size >  1.25)[0]
                score_small = metrics.cohen_kappa_score(
                    np.array(trues_list)[small_idx],
                    np.array(preds_list)[small_idx],
                    weights='quadratic')
                score_medium = metrics.cohen_kappa_score(
                    np.array(trues_list)[medium_idx],
                    np.array(preds_list)[medium_idx],
                    weights='quadratic')
                score_large = metrics.cohen_kappa_score(
                    np.array(trues_list)[large_idx],
                    np.array(preds_list)[large_idx],
                    weights='quadratic')
            else:
                score_small = None
                score_medium = None
                score_large = None

            if data_provider is not None:
                karolinska_idx = np.where(data_provider == 'karolinska')[0]
                radboud_idx = np.where(data_provider == 'radboud')[0]
                score_karolinska = metrics.cohen_kappa_score(
                    np.array(trues_list)[karolinska_idx],
                    np.array(preds_list)[karolinska_idx],
                    weights='quadratic')
                score_radboud = metrics.cohen_kappa_score(
                    np.array(trues_list)[radboud_idx],
                    np.array(preds_list)[radboud_idx],
                    weights='quadratic')
            else:
                score_karolinska = None
                score_radboud = None

            score = metrics.cohen_kappa_score(
                trues_list, preds_list, weights='quadratic')

            if score > best_score:
                best_score = score
                self.keras_model.save_weights(f'output/weights/model-{fold}-{epoch}.h5')

            with open('output/scores.txt', 'a') as f:
                if epoch == 0:
                    f.write(
                        'fold'              + ',' +
                        'epoch'             + ',' +
                        'score'             + ',' +
                        'score_small'       + ',' +
                        'score_medium'      + ',' +
                        'score_large'       + ',' +
                        'score_karolinska'  + ',' +
                        'score_radboud'     + '\n'
                    )
                f.write(
                    str(fold)              + ',' +
                    str(epoch)         + ',' +
                    str(score)             + ',' +
                    str(score_small)       + ',' +
                    str(score_medium)      + ',' +
                    str(score_large)       + ',' +
                    str(score_karolinska)  + ',' +
                    str(score_radboud)     + '\n'
                )
