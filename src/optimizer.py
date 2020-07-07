import tensorflow as tf


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
