import numpy as np
import pandas as pd
import multiprocessing
import tqdm
import cv2
import skimage.io
import math
import os

from config import Config
from generator import read_image, resize_image


def mask_special(image):
    # Define elliptic kernel
    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    image_red = image[:, :, 0].reshape(-1)
    image_red = image_red[np.where(image_red < 220)[0]]
    image_red_mean = np.mean(image_red)
    image_red_std = np.std(image_red)

    # use cv2.inRange to mask pen marks (hardcoded for now)
    lower = np.array([0, 0, 0])
    upper = np.array([int(image_red_mean-image_red_std//4), 255, 255])
    img_mask1 = cv2.inRange(image, lower, upper)

    # Use erosion and findContours to remove masked tissue (side effect of above)
    img_mask1 = cv2.erode(img_mask1, kernel5x5, iterations=4)
    img_mask2 = np.zeros(img_mask1.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(img_mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        w, h = x.max()-x.min(), y.max()-y.min()
        if w > 128 or h > 128:
            cv2.drawContours(img_mask2, [contour], 0, 1, -1)
    # expand the area of the pen marks
    img_mask2 = cv2.dilate(img_mask2, kernel5x5, iterations=5)
    return img_mask2

def mask_image(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_mask = np.where(image_gray < 220, 1, 0).astype(np.uint8)

    image_mask = cv2.dilate(image_mask, kernel, iterations=1)
    contour, _ = cv2.findContours(image_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(image_mask, [cnt], 0, 1, -1)
    return image_mask


def compute_coords(image,
                   image_mask,
                   image_mask2=None,
                   patch_size=Config.IMAGE_PATCH_SIZE,
                   alpha=0.4,
                   beta=0.5,
                   gamma=0.25,
                   delta=0.7):

    """
    Hyperparameters:
        alpha: Final threshold, the minimum information in patch
            - formula: if mask_patch.mean() > alpha: keep
        beta: Minimum number of on-bits in x/y dimension to be considered enough information at y/x index.
            - formula: patch_size * beta
        gamma: Minimum consecutive y/x indices from beta
            - formula: patch_size * gamma
        delta: Threshold for decimal point for removing one patch in y/x dim
            - formula: if number_of_patches % 1 < delta: remove

    """


    # initialize accumulator
    coords = np.zeros([0, 3], dtype=int)

    # Pad image to not miss out on literal edge cases
    image_mask = np.pad(
        image_mask, ((patch_size, patch_size), (patch_size, patch_size)), 'maximum')
    image = np.pad(
        image, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), 'maximum')
    if image_mask2 is not None:
        image_mask2 = np.pad(
            image_mask2, ((patch_size, patch_size), (patch_size, patch_size)), 'minimum')


    # test if tissue is align vertically or horizontally
    # if horizontal, transpose it
    y_sum = np.sum(image_mask, axis=1)
    x_sum = np.sum(image_mask, axis=0)
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        transposed = True
        image_mask = image_mask.T
        if image_mask2 is not None:
            image_mask2 = image_mask2.T
        image = np.transpose(image, (1, 0, 2))
        y_sum, x_sum = x_sum, y_sum
    else:
        transposed = False

    y_indices = np.where(y_sum >= (patch_size*beta))[0]

    if len(y_indices) < 1:
        print("FAIL AT 1")
        return [(0, 0, 0)]

    y_indices = np.split(y_indices, np.where(np.diff(y_indices) != 1)[0]+1)
    y_indices = [y_index for y_index in y_indices if len(y_index) >= patch_size*gamma]

    if len(y_indices) < 1:
        print("FAIL AT 2")
        return [(0, 0, 0)]

    for y_index in y_indices:
        y_start, y_end = y_index[0], y_index[-1]

        y_num_slices = (y_end-y_start)/patch_size

        if y_num_slices % 1 < delta and y_num_slices >= 1:
            y_num_slices = math.floor(y_num_slices)
        else:
            y_num_slices = math.ceil(y_num_slices)

        y_excess = (y_num_slices*patch_size) - (y_end-y_start)
        y_shift = y_excess // 2


        for i in range(y_num_slices):
            y = i * patch_size + y_start - y_shift
            x_sub_sum = image_mask[y:y+patch_size, :].sum(axis=0)
            x_indices = np.where(x_sub_sum >= (patch_size*beta))[0]

            x_indices = np.split(x_indices, np.where(np.diff(x_indices) != 1)[0]+1)
            x_indices = [x_index for x_index in x_indices if len(x_index) >= patch_size*gamma]

            for x_index in x_indices:
                if len(x_index) >= 1:
                    x_start, x_end = x_index[0], x_index[-1]
                    x_num_slices = (x_end-x_start)/patch_size
                    if x_num_slices % 1 < delta and x_num_slices >= 1:
                        x_num_slices = math.floor(x_num_slices)
                    else:
                        x_num_slices = math.ceil(x_num_slices)
                    x_excess = (x_num_slices * patch_size) - (x_end-x_start)
                    x_shift = x_excess // 2
                    for j in range(x_num_slices):
                        x = j * patch_size + x_start-x_shift


                        if image_mask2 is not None:
                            val0 = image_mask2[y: y+patch_size, x:x+patch_size].sum()
                        else:
                            val0 = 0

                        patch_1d = image[y: y+patch_size, x:x+patch_size, :].reshape(-1)
                        idx = np.where(patch_1d < 220)[0]
                        if len(idx) > 0 and val0 < (patch_size*5):
                            val1 = int(patch_1d[idx].mean())
                            val2 = image_mask[y:y+patch_size, x:x+patch_size].mean()
                            if val2 > alpha:
                                if transposed:
                                    coords = np.concatenate([
                                        coords, [[val1, x-patch_size, y-patch_size]]
                                    ])
                                else:
                                    coords = np.concatenate([
                                        coords, [[val1, y-patch_size, x-patch_size]]
                                    ])
    if len(coords) < 1:
        print("FAIL AT 3")
        return [(0, 0, 0)]

    return coords

if os.path.isfile('output/marked_images.npy'):
    marked_images = np.load('output/marked_images.npy', allow_pickle=True)

    def compose(image_path):
        image = read_image(image_path)
        image = resize_image(image)
        mask = mask_image(image)

        ID = image_path.split('/')[-1].split('.')[0]
        if ID in marked_images:
            special_mask = mask_special(image)
            coords = compute_coords(image, mask, special_mask)
        else:
            coords = compute_coords(image, mask, None)
        return coords
else:
    def compose(image_path):
        image = read_image(image_path)
        image = resize_image(image)
        mask = mask_image(image)
        coords = compute_coords(image, mask, None)
        return coords


def compute(data, image_path, save_to_file=False):
    paths = image_path + data.image_id + '.tiff'
    coords = []
    with multiprocessing.Pool() as pool:
        for c in tqdm.tqdm(pool.imap(compose, paths), total=len(paths)):
            coords.append(c)
    coords = np.array(coords)
    if save_to_file:
        np.save('output/coordinates.npy', coords)
    return coords

if __name__ == "__main__":
    path = '../../input/prostate-cancer-grade-assessment/'
    data = pd.read_csv(path+'train.csv')
    _ = compute(data, path+'train_images/', save_to_file=True)
