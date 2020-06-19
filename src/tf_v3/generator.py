import tensorflow as tf
import numpy as np
import pandas as pd
import math
import cv2
import skimage.io
import glob
import os
import io
import warnings
from PIL import Image
Image.MAX_IMAGE_PIXELS = 20000000000

from albumentations import *
import albumentations.augmentations.functional as F


from config import Config

def map_decorator(func):
    def wrapper(*args):
        return tf.py_function(
            func=func,
            inp=[*args],
            Tout=[tf.uint8, tf.int32]
        )
    return wrapper

augmentor = (
    Compose([
      # OneOf([
      #   ShiftScaleRotate(
      #       shift_limit=0.0625,
      #       scale_limit=0.1,
      #       rotate_limit=90,
      #       p=0.5),
      #   ElasticTransform(
      #       alpha=601,
      #       sigma=25,
      #       alpha_affine=10,
      #       p=0.5),
      #   GridDistortion(
      #       num_steps=3,
      #       distort_limit=0.5,
      #       p=0.5),
      #   NoOp()
      # ]),
      # OneOf([
      #   RandomBrightnessContrast(
      #       brightness_limit=0.15,
      #       contrast_limit=0.15,
      #       p=0.5),
      #   RandomGamma(
      #       gamma_limit=(85, 115),
      #       p=0.5),
      #   NoOp()
      # ]),
      # OneOf([
      #   RGBShift(
      #       r_shift_limit=(-15, 15),
      #       g_shift_limit=(-15, 15),
      #       b_shift_limit=(-15, 15),
      #       p=0.5),
      #   HueSaturationValue(
      #       hue_shift_limit=(-15, 15),
      #       sat_shift_limit=(-25, 25),
      #       val_shift_limit=(-15, 15),
      #       p=0.5),
      #   NoOp()
      # ]),
      RandomRotate90(
          p=0.5),
      Flip(
          p=0.5),
      Transpose(
          p=0.5),
   ])
)

@map_decorator
def augmentation(image, label):
    image = augmentor(image=image.numpy())['image']
    return image, label

def read_image(image_path, label, resize_ratio=Config.input.resize_ratio):
    if not(isinstance(image_path, str)):
        image_path = image_path.numpy().decode('utf-8')
    image = Image.open(image_path)
    image = np.array(image)
    if resize_ratio != 1:
        new_w = int(image.shape[1]*resize_ratio)
        new_h = int(image.shape[0]*resize_ratio)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, label

def _mask_tissue(image, kernel_size=(7, 7), gray_threshold=220):
    """Masks tissue in image. Uses gray-scaled image, as well as
    dilation kernels and 'gap filling'
    """
    # Define elliptic kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Convert rgb to gray scale for easier masking
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Now mask the gray-scaled image (capturing tissue in biopsy)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    # Use dilation and findContours to fill in gaps/holes in masked tissue
    mask = cv2.dilate(mask, kernel, iterations=1)
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 1, -1)
    return mask

def _pad_image(image, pad_len, pad_val):
    """Pads inputted image, accepts both
    2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
    elif image.ndim == 3:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
    return None

def _transpose_image(image):
    """Inputs an image and transposes it, accepts
    both 2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.transpose(image, (1, 0)).copy()
    elif image.ndim == 3:
        return np.transpose(image, (1, 0, 2)).copy()
    return None

def _get_tissue_parts_indices(tissue, min_consec_info):
    """If there are multiple tissue parts in 'tissue', 'tissue' will be
    split it. Each tissue part will be taken care of separately, and if
    the tissue part is less than min_consec_info, it's considered to small
    and won't be returned.
    """
    split_points = np.where(np.diff(tissue) != 1)[0]+1
    tissue_parts = np.split(tissue, split_points)
    return [
        tp for tp in tissue_parts if len(tp) >= min_consec_info
    ]

def _get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
    This tissue part is divided into N subparts and returned.
    Argument min_decimal_keep basically decides if we should squeeze in the
    N subparts in an area bigger than the sum of the N subparts or not.
    """
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end-start)/patch_size
    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts*patch_size) - (end-start)
    shift = excess // 2

    return [
        i * patch_size + start - shift
        for i in range(num_subparts)
    ]

def _eval_and_append_xy_coords(coords,
                               image,
                               mask,
                               patch_size,
                               x, y,
                               min_patch_info,
                               transposed,
                               precompute):
    """Based on computed x and y coordinate of patch,
    slices out patch from original image, flattens it,
    preprocesses it, and finally evaluates its mask.
    If patch contains more info than min_patch_info,
    the patch coordinates are kept, along with a
    value ('val1') that estimates how much information
    there is in the patch.
    """
    patch_1d = (
        image[y: y+patch_size, x:x+patch_size, :]
        .mean(axis=2)
        .reshape(-1)
    )
    idx_tissue = np.where(patch_1d <= 210)[0]
    idx_black = np.where(patch_1d < 5)[0]
    idx_background = np.where(patch_1d > 210)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = 210
        patch_1d[idx_background] = 210
        val1 = int(patch_1d.mean())
        val2 = mask[y:y+patch_size, x:x+patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([
                        coords, [[val1, x-patch_size, y-patch_size]]
                    ])
                else:
                    coords = np.concatenate([
                        coords, [[val1, y-patch_size, x-patch_size]]
                    ])
            else:
                coords = np.concatenate([
                    coords, [[val1, y, x]]
                ])

    return coords

def compute_coords(image,
                   label,
                   patch_size=Config.input.patch_size,
                   min_patch_info=0.35,
                   min_axis_info=0.35,
                   min_consec_axis_info=0.35,
                   min_decimal_keep=0.7):

    """
    Input:
        image : np.ndarray (RGB image)
        patch_size : size of patches/tiles, will be of
            size (patch_size x patch_size x 3)
        precompute : If True, only coordinates will be returned,
            these coordinates match the inputted 'original' image.
            If False, both an image and coordinates will be returned,
            the coordinates does not match the inputted image but the
            image that it is returned with.
        min_patch_info : Minimum required information in patch
            (see '_eval_and_append_xy_coords')
        min_axis_info : Minimum fraction of on-bits in x/y dimension to be
            considered enough information. For x, this would be fraction of
            on-bits in x-dimension of a y:y+patch_size slice. For y, this would
            be the fraction of on-bits for the whole image in y-dimension
        min_consec_axis_info : Minimum consecutive x/y on-bits from 'min_axis_info'
            (see '_get_tissue_parts_indices')
        min_decimal_keep : Threshold for decimal point for removing "excessive" patch
            (see '_get_tissue_subparts_coords')

    Output:
        image [only if precompute is True] : similar to input image, but fits
            to the computed coordinates
        coords : the coordinates that will be used to compute the patches later on
    """


    if type(image) != np.ndarray:
        image = image.numpy()

    # masked tissue will be used to compute the coordinates
    mask = _mask_tissue(image)

    # initialize coordinate accumulator
    coords = np.zeros([0, 3], dtype=int)

    # pad image and mask to make sure no tissue is potentially missed out
    image = _pad_image(image, patch_size, 'maximum')
    mask = _pad_image(mask, patch_size, 'minimum')

    y_sum = mask.sum(axis=1)
    x_sum = mask.sum(axis=0)
    # if on bits in x_sum is greater than in y_sum, the tissue is
    # likely aligned horizontally. The algorithm works better if
    # the image is aligned vertically, thus the image will be transposed
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        image = _transpose_image(image)
        mask = _transpose_image(mask)
        y_sum, _ = x_sum, y_sum
        transposed = True
    else:
        transposed = False

    # where y_sum is more than the minimum number of on-bits
    y_tissue = np.where(y_sum >= (patch_size*min_axis_info))[0]

    if len(y_tissue) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        return image, label, [(0, 0, 0)]

    y_tissue_parts_indices = _get_tissue_parts_indices(
        y_tissue, patch_size*min_consec_axis_info)

    if len(y_tissue_parts_indices) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        return image, label, [(0, 0, 0)]

    # loop over the tissues in y-dimension
    for yidx in y_tissue_parts_indices:
        y_tissue_subparts_coords = _get_tissue_subparts_coords(
            yidx, patch_size, min_decimal_keep)

        for y in y_tissue_subparts_coords:
            # in y_slice, where x_slice_sum is more than the minimum number of on-bits
            x_slice_sum = mask[y:y+patch_size, :].sum(axis=0)
            x_tissue = np.where(x_slice_sum >= (patch_size*min_axis_info))[0]

            x_tissue_parts_indices = _get_tissue_parts_indices(
                x_tissue, patch_size*min_consec_axis_info)

            # loop over tissues in x-dimension (inside y_slice 'y:y+patch_size')
            for xidx in x_tissue_parts_indices:
                x_tissue_subparts_coords = _get_tissue_subparts_coords(
                    xidx, patch_size, min_decimal_keep)

                for x in x_tissue_subparts_coords:
                    coords = _eval_and_append_xy_coords(
                        coords, image, mask, patch_size, x, y,
                        min_patch_info, transposed, precompute=False,
                    )

    if len(coords) < 1:
        warnings.warn("Not enough tissue in image (x-dim)", RuntimeWarning)
        return image, label, [(0, 0, 0)]

    return image, label, coords


def _patch_augment(patch):
    """Performs random rotation, random flip (u/d, l/r),
    and random transpose, based on probability p"""
    p = 0.5
    r1 = tf.random.uniform(
        shape=(4,), minval=0, maxval=1)
    r2 = tf.random.uniform(
        shape=(), minval=0, maxval=4, dtype=tf.int32)
    if r1[0] > (1-p):
        patch = tf.image.rot90(patch, k=r2)
    if r1[1] > (1-p):
        patch = tf.image.random_flip_left_right(patch)
    if r1[2] > (1-p):
        patch = tf.image.random_flip_up_down(patch)
    if r1[3] > (1-p):
        patch = tf.transpose(patch, (1, 0, 2))
    return patch

def _coord_excess_filtering(coords, sample_size):
    """filters out a portion of excessive coordinates.
    coordinates with higher values are filtered out.
    """
    if len(coords) > sample_size:
        c = tf.transpose(coords)
        v = tf.gather(c, 0)
        num = tf.cast(len(v), tf.float32)
        sample_size = tf.cast(sample_size, tf.float32)
        downscale = int(tf.math.ceil(
            num * (1 - ((num - sample_size) / num) * 0.25)
        ))
        v_argsort = tf.argsort(v)
        indices = tf.gather(v_argsort, tf.range(downscale))
        indices = tf.sort(indices)
        coords = tf.gather(coords, indices)
    return coords

def patch_image(image, label, coords,
                sample_size=Config.input.sample_size,
                patch_size=config.input.patch_size):
    """Turns image into patches, and stitches it together.
    Patches are selected based on coords, which have been
    precomputed"""

    l = tf.cast(tf.math.sqrt(tf.cast(sample_size, tf.float32)), tf.int32)

    #coords = _coord_padding_removal(coords)
    coords = _coord_excess_filtering(coords, sample_size)
    coords = tf.random.shuffle(coords)
    if len(coords) < sample_size:
        indices = tf.tile(
            tf.range(len(coords)), [tf.math.ceil(sample_size/len(coords))])
        indices = indices[:sample_size]
    else:
        indices = tf.range(sample_size)

    coords = tf.gather(coords, indices)

    patched_image = tf.zeros(
        [0, patch_size, patch_size, 3], dtype=tf.dtypes.uint8)

    for i in range(sample_size):
        y = tf.gather_nd(coords, [i, 1])
        x = tf.gather_nd(coords, [i, 2])
        shape = tf.shape(image)
        h = tf.gather(shape, 0)
        w = tf.gather(shape, 1)
        if y < 0: y = 0
        if x < 0: x = 0
        if y > h-patch_size: y = h-patch_size
        if x > w-patch_size: x = w-patch_size

        patch = tf.slice(
            image,
            tf.stack([y, x, 0]),
            tf.stack([patch_size, patch_size, -1]))

        patch = _patch_augment(patch)
        patched_image = tf.concat([
            patched_image, tf.expand_dims(patch, 0)], axis=0)

    patched_image = tf.reshape(patched_image, (-1, patch_size*l, patch_size, 3))
    patched_image = tf.transpose(patched_image, (0, 2, 1, 3))
    patched_image = tf.reshape(patched_image, (patch_size*l, patch_size*l, 3))

    return patched_image, label

@tf.function
def preprocess_input(x, y, mode=Config.input.preprocess_mode):
    x = tf.cast(x, dtype=tf.dtypes.float32)
    y = tf.cast(y, dtype=tf.dtypes.float32)
    #x = 255. - x
    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x, y
    elif mode == 'torch':
        x /= 255.
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        x -= mean
        x /= std
        return x, y
    elif mode == 'caffe':
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x -= mean
        return x, y
    elif mode == 'float':
        x /= 255.
        return x, y
    else:
        return x, y

def get_dataset(dataframe,
                input_path,
                batch_size,
                training,
                augment,
                tta=1,
                input_size=(1536, 1536, 3),
                buffer_size=1,
                cache=False):


    if cache:
        if not(os.path.isdir('../tmp/')):
            os.mkdir('../tmp')
        else:
            files = glob.glob(f'../tmp/*')
            for f in files:
                os.remove(f)
        if training:
            cache_path = '../tmp/cache_train'
        else:
            cache_path = '../tmp/cache_infer'

    image_paths = (input_path + dataframe.image_id + '.jpeg').values.astype(str)
    labels = np.zeros((len(dataframe), 5), dtype=np.int32)
    for i, label in enumerate(dataframe.isup_grade):
        labels[i, :label] = 1.

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(
    lambda x, y: tf.py_function(
            func=read_image,
            inp=[x, y],
            Tout=[tf.uint8, tf.int32]),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x, y: tf.py_function(
            func=compute_coords,
            inp=[x, y],
            Tout=[tf.uint8, tf.int32, tf.int32]
        ),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(patch_image, tf.data.experimental.AUTOTUNE)

    if cache:
        dataset = dataset.cache(cache_path)

    if augment:
        dataset = dataset.map(augmentation, tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess_input, tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
