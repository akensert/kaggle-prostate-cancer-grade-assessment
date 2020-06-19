"""
Reference:
    https://github.com/titu1994/keras_mixnets
All credit to:
    titu1994; Thank you very much!
This script:
    Slightly modified from the original. Also, all scripts are put into one.
"""

import math
import re
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import initializers
from tensorflow.keras.utils import get_custom_objects


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class MixNetConvInitializer(initializers.Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    # Arguments:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    # Returns:
      an initialization for the variable
    """
    def __init__(self):
        super(MixNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random.normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class MixNetDenseInitializer(initializers.Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                          distribution='uniform').
        It is written out explicitly base_path for clarity.
        # Arguments:
          shape: shape of variable
          dtype: dtype of variable
          partition_info: unused
        # Returns:
          an initialization for the variable
    """
    def __init__(self):
        super(MixNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class Swish(layers.Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class DropConnect(layers.Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {
            'drop_connect_rate': self.drop_connect_rate,
        }
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GroupConvolution(layers.Layer):

    def __init__(self, filters, kernels, groups,
                 type='conv', conv_kwargs=None,
                 **kwargs):
        super(GroupConvolution, self).__init__(**kwargs)

        if conv_kwargs is None:
            conv_kwargs = {
                'strides': (1, 1),
                'padding': 'same',
                'dilation_rate': (1, 1),
                'use_bias': False,
            }

        self.filters = filters
        self.kernels = kernels
        self.groups = groups
        self.type = type
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}

        assert type in ['conv', 'depthwise_conv']
        if type == 'conv':
            splits = self._split_channels(filters, self.groups)
            self._layers = [layers.Conv2D(splits[i], kernels[i],
                                          strides=self.strides,
                                          padding=self.padding,
                                          dilation_rate=self.dilation_rate,
                                          use_bias=self.use_bias,
                                          kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]

        else:
            self._layers = [layers.DepthwiseConv2D(kernels[i],
                                                   strides=self.strides,
                                                   padding=self.padding,
                                                   dilation_rate=self.dilation_rate,
                                                   use_bias=self.use_bias,
                                                   kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]

        self.data_format = 'channels_last'
        self._channel_axis = -1

    def call(self, inputs, **kwargs):
        if len(self._layers) == 1:
            return self._layers[0](inputs)

        filters = K.int_shape(inputs)[self._channel_axis]
        splits = self._split_channels(filters, self.groups)
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._layers)]
        x = layers.concatenate(x_outputs, axis=self._channel_axis)
        return x

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                filter_size=1,
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernels': self.kernels,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'type': self.type,
            'conv_kwargs': self.conv_kwargs,
        }
        base_config = super(GroupConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BlockArgs(object):

    def __init__(self, input_filters=None,
                 output_filters=None,
                 dw_kernel_size=None,
                 expand_kernel_size=None,
                 project_kernel_size=None,
                 strides=None,
                 num_repeat=None,
                 se_ratio=None,
                 expand_ratio=None,
                 identity_skip=True,
                 swish=False,
                 dilated=False):

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.dw_kernel_size = self._normalize_kernel_size(dw_kernel_size)
        self.expand_kernel_size = self._normalize_kernel_size(expand_kernel_size)
        self.project_kernel_size = self._normalize_kernel_size(project_kernel_size)
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip
        self.swish = swish
        self.dilated = dilated

    def decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        self.input_filters = int(options['i'])
        self.output_filters = int(options['o'])
        self.dw_kernel_size = self._parse_ksize(options['k'])
        self.expand_kernel_size = self._parse_ksize(options['a'])
        self.project_kernel_size = self._parse_ksize(options['p'])
        self.num_repeat = int(options['r'])
        self.identity_skip = ('noskip' not in block_string)
        self.se_ratio = float(options['se']) if 'se' in options else None
        self.expand_ratio = int(options['e'])
        self.strides = [int(options['s'][0]), int(options['s'][1])]
        self.swish = 'sw' in block_string
        self.dilated = 'dilated' in block_string

        return self

    def encode_block_string(self, block):
        """Encodes a block to a string.
        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments
        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```
        """

        args = [
            'r%d' % block.num_repeat,
            'k%s' % self._encode_ksize(block.kernel_size),
            'a%s' % self._encode_ksize(block.expand_kernel_size),
            'p%s' % self._encode_ksize(block.project_kernel_size),
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]

        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)

        if block.id_skip is False:
            args.append('noskip')

        if block.swish:
            args.append('sw')

        if block.dilated:
            args.append('dilated')

        return '_'.join(args)

    def _normalize_kernel_size(self, val):
        if type(val) == int:
            return [val]

        return val

    def _parse_ksize(self, ss):
        return [int(k) for k in ss.split('.')]

    def _encode_ksize(self, arr):
        return '.'.join([str(k) for k in arr])

    @classmethod
    def from_block_string(cls, block_string):
        """
        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments
        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```
        Returns:
            BlockArgs object initialized with the block
            string args.
        """
        block = cls()
        return block.decode_block_string(block_string)


# Default list of blocks for MixNets
def get_mixnet_small(depth_multiplier=None):
    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i16_o16',
        'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
        'r1_k3_a1.1_p1.1_s11_e3_i24_o24',

        'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5_sw',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

        'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25_sw',
        'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25_sw',

        'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5_sw',
        'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

        'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5_sw',
        'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    ]
    DEFAULT_BLOCK_LIST = [BlockArgs.from_block_string(s)
                          for s in blocks_args]

    return DEFAULT_BLOCK_LIST


def get_mixnet_medium(depth_multiplier=None):
    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i24_o24',
        'r1_k3.5.7_a1.1_p1.1_s22_e6_i24_o32',
        'r1_k3_a1.1_p1.1_s11_e3_i32_o32',

        'r1_k3.5.7.9_a1_p1_s22_e6_i32_o40_se0.5_sw',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

        'r1_k3.5.7_a1_p1_s22_e6_i40_o80_se0.25_sw',
        'r3_k3.5.7.9_a1.1_p1.1_s11_e6_i80_o80_se0.25_sw',

        'r1_k3_a1_p1_s11_e6_i80_o120_se0.5_sw',
        'r3_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

        'r1_k3.5.7.9_a1_p1_s22_e6_i120_o200_se0.5_sw',
        'r3_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    ]

    DEFAULT_BLOCK_LIST = [BlockArgs.from_block_string(s)
                          for s in blocks_args]

    return DEFAULT_BLOCK_LIST


def get_mixnet_large(depth_multiplier=None):
    return get_mixnet_medium(depth_multiplier)


__all__ = ['MixNet',
           'MixNetSmall',
           'MixNetMedium',
           'MixNetLarge',
           'preprocess_input']

GROUP_NUM = 1


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
def _split_channels(total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def round_filters(filters, depth_multiplier, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(depth_multiplier) if depth_multiplier is not None else None
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def round_repeats(repeats):
    """Round number of filters based on depth multiplier."""
    return int(repeats)


# Ontained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
class GroupedConv2D(object):
    """Groupped convolution.
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel_size size.
    """

    def __init__(self, filters, kernel_size, **kwargs):
        """Initialize the layer.
        Args:
          filters: Integer, the dimensionality of the output space.
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel_size for each group.
          **kwargs: other parameters passed to the original conv2d layer.
        """

        global GROUP_NUM
        self._groups = len(kernel_size)
        self._channel_axis = -1
        self.filters = filters
        self.kernels = kernel_size

        self._conv_kwargs = {
            'strides': kwargs.get('strides', (1, 1)),
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
        }

        GROUP_NUM += 1

    def __call__(self, inputs):
        grouped_op = GroupConvolution(self.filters, self.kernels, groups=self._groups,
                                      type='conv', conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, activation_fn, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = GroupedConv2D(
            num_reduced_filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)

        x = activation_fn()(x)

        # Excite
        x = GroupedConv2D(
            filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = layers.Activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block


# Obtained from
class MDConv(object):
    """MDConv with mixed depthwise convolutional kernels.
    MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
    3x3, 5x5, etc). Right now, we use an naive implementation that split channels
    into multiple groups and perform different kernels for each group.
    See Mixnet paper for more details.
    """

    def __init__(self, kernel_size, strides, dilated=False, **kwargs):
        """Initialize the layer.
        Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
        an extra parameter "dilated" to indicate whether to use dilated conv to
        simulate large kernel_size size. If dilated=True, then dilation_rate is ignored.
        Args:
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
            then we split the channels and perform different kernel_size for each group.
          strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width.
          dilated: Bool. indicate whether to use dilated conv to simulate large
            kernel_size size.
          **kwargs: other parameters passed to the original depthwise_conv layer.
        """
        self._channel_axis = -1
        self._dilated = dilated
        self.kernels = kernel_size

        self._conv_kwargs = {
            'strides': strides,
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
        }

    def __call__(self, inputs):
        filters = K.int_shape(inputs)[self._channel_axis]
        grouped_op = GroupConvolution(filters, self.kernels, groups=len(self.kernels),
                                      type='depthwise_conv', conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def MixNetBlock(input_filters, output_filters,
                dw_kernel_size, expand_kernel_size,
                project_kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                swish=False,
                dilated=None,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio
    relu_activation = Swish if swish else layers.ReLU

    def block(inputs):
        # Expand part
        if expand_ratio != 1:
            x = GroupedConv2D(
                filters,
                kernel_size=expand_kernel_size,
                strides=[1, 1],
                kernel_initializer=MixNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)

            x = layers.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)

            x = relu_activation()(x)
        else:
            x = inputs

        kernel_size = dw_kernel_size
        # Depthwise Convolutional Phase
        x = MDConv(
            kernel_size,
            strides=strides,
            dilated=dilated,
            depthwise_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = relu_activation()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        relu_activation,
                        data_format)(x)

        # output phase
        x = GroupedConv2D(
            output_filters,
            kernel_size=project_kernel_size,
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                # if drop_connect_rate:
                #     x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])

        return x

    return block


def MixNet(input_shape,
           block_args_list,
           depth_multiplier,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000,
           dropout_rate=0.,
           drop_connect_rate=0.,
           batch_norm_momentum=0.99,
           batch_norm_epsilon=1e-3,
           depth_divisor=8,
           stem_size=16,
           feature_size=1536,
           min_depth=None,
           data_format=None,
           default_size=None,
           **kwargs):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_mixnet_small()

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)


    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    # Stem part
    x = inputs
    x = GroupedConv2D(
        filters=round_filters(stem_size, depth_multiplier,
                              depth_divisor, min_depth),
        kernel_size=[3],
        strides=[2, 2],
        kernel_initializer=MixNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = layers.ReLU()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, depth_multiplier, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, depth_multiplier, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat)

        # The first block needs to take care of stride and filter size increase.
        x = MixNetBlock(block_args.input_filters, block_args.output_filters,
                        block_args.dw_kernel_size, block_args.expand_kernel_size,
                        block_args.project_kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                        block_args.dilated, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MixNetBlock(block_args.input_filters, block_args.output_filters,
                            block_args.dw_kernel_size, block_args.expand_kernel_size,
                            block_args.project_kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                            block_args.dilated, data_format)(x)

    # Head part
    x = GroupedConv2D(
        filters=feature_size,
        kernel_size=[1],
        strides=[1, 1],
        kernel_initializer=MixNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = layers.ReLU()(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(classes, kernel_initializer=MixNetDenseInitializer())(x)
        x = layers.Activation('softmax')(x)

    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    outputs = x

    model = Model(inputs, outputs)

    return model


def MixNetSmall(input_shape=None,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
                dropout_rate=0.2,
                drop_connect_rate=0.,
                data_format=None):

    return MixNet(input_shape,
                  get_mixnet_small(),
                  depth_multiplier=1.0,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  data_format=data_format,
                  default_size=224)


def MixNetMedium(input_shape=None,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.25,
                 drop_connect_rate=0.,
                 data_format=None):

    return MixNet(input_shape,
                  get_mixnet_medium(),
                  depth_multiplier=1.0,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  stem_size=24,
                  data_format=data_format,
                  default_size=224)


def MixNetLarge(input_shape=None,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
                dropout_rate=0.3,
                drop_connect_rate=0.,
                data_format=None):

    return MixNet(input_shape,
                  get_mixnet_large(),
                  depth_multiplier=1.3,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  stem_size=24,
                  data_format=data_format,
                  default_size=224)
