import tensorflow as tf
import numpy as np
import pandas as pd
import os


def get_optimizer(steps_per_epoch, lr_max, lr_min,
                  decay_epochs, warmup_epochs, power=1):

    if decay_epochs > 0:
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr_max,
            decay_steps=steps_per_epoch*decay_epochs,
            end_learning_rate=lr_min,
            power=power,
        )
    else:
        learning_rate_fn = lr_max

    if warmup_epochs > 0:
        learning_rate_fn = WarmUp(
            lr_start = lr_min,
            lr_end = lr_max,
            lr_fn = learning_rate_fn,
            warmup_steps=steps_per_epoch*warmup_epochs,
            power=power,
        )

    return tf.keras.optimizers.Adam(learning_rate_fn)


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_start, lr_end, lr_fn, warmup_steps, power=1):
        super().__init__()
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_fn = lr_fn
        self.warmup_steps = warmup_steps
        self.power = power

    def __call__(self, step):
        global_step_float = tf.cast(step, tf.float32)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
        warmup_percent_done = global_step_float / warmup_steps_float
        warmup_learning_rate = tf.add(tf.multiply(
            self.lr_start-self.lr_end,
            tf.math.pow(1-warmup_percent_done, self.power)), self.lr_end)
        return tf.cond(
            global_step_float < warmup_steps_float,
            lambda: warmup_learning_rate,
            lambda: self.lr_fn(step),
        )

    def get_config(self):
        return {
            "lr_start": self.lr_start,
            "lr_end": self.lr_end,
            "lr_fn": self.lr_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
        }


class DataManager:
    """This class is created only for convenience --- and is specific to
    the PANDA [Kaggle challenge] dataset. It can directly be used locally,
    on a server, or in a Kaggle kernel

    Example usage:
        train_data, test_data, sub_data = DataManager.get_all_files()
    """
    @staticmethod
    def get_path():
        directory = 'input/prostate-cancer-grade-assessment/'
        navigate = ''
        while 1:
            if os.path.isdir(navigate + directory):
                path = navigate + directory
                break
            navigate += '../'
            if len(navigate) > 15:
                print("Directory {} not found".format(directory))
                path = None
                break
        return path

    @classmethod
    def get_train_file(cls):
        if os.path.isfile(cls.get_path() + 'train.csv'):
            return pd.read_csv(cls.get_path() + 'train.csv')
        return None

    @classmethod
    def get_test_file(cls):
        if os.path.isfile(cls.get_path() + 'test.csv'):
            return pd.read_csv(cls.get_path() + 'test.csv')
        return None

    @classmethod
    def get_submission_file(cls):
        if os.path.isfile(cls.get_path() + 'sample_submission.csv'):
            return pd.read_csv(cls.get_path() + 'sample_submission.csv')
        return None

    @classmethod
    def get_all_files(cls):
        return (
            cls.get_train_file(),
            cls.get_test_file(),
            cls.get_submission_file()
        )

if __name__ == "__main__":
    
    train_data, test_data, sub_data = DataManager.get_all_files()

    if train_data is not None:
        print("train_data shape   =", train_data.shape)
        print("train_data columns =", train_data.columns.values)
        print("train_data dtypes  =", train_data.dtypes.values)
    else:
        print("train.csv not found")

    if test_data is not None:
        print("test_data shape    =", test_data.shape)
        print("test_data columns  =", test_data.columns.values)
        print("test_data dtypes   =", test_data.dtypes.values)
    else:
        print("test.csv not found")

    if sub_data is not None:
        print("sub_data shape     =", sub_data.shape)
        print("sub_data columns   =", sub_data.columns.values)
        print("sub_data dtypes    =", sub_data.dtypes.values)
    else:
        print("submission.csv not found")
