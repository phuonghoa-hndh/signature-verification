import itertools
import os
import random

import cv2
import numpy as np
from sklearn.utils import shuffle

import signature_verification.constants as cons


def get_groups(path, idx_writers_file):
    """
    This function is to get origin and forgery signatures in a folder which has the following structure:
        - 1:
            + forgeries_1_1.png
            + forgeries_1_2.png
            ...
            + original_1_1.png
            + original_1_2.png
            ...
        - 2:
            + ...
    Args:
        path: path to the folder which stores data
        idx_writers_file: txt file that has the number of writer
    Returns:
        orig_groups, forg_groups: list of origin and forgery signatures
    """
    with open(idx_writers_file) as f:
        content = f.read()
        idx_writers_file = content.strip().split(",")

    orig_groups, forg_groups = [], []
    for idx_writer in idx_writers_file:
        images = os.listdir(path + "/" + idx_writer)
        images.sort()
        images = [path + "/" + idx_writer + "/" + x for x in images]
        forg_groups.append(images[:24])  # First 24 signatures in each folder are forged
        orig_groups.append(images[24:])  # Next 24 signatures are genuine

    return orig_groups, forg_groups


def generate_batch(orig_groups, forg_groups, batch_size=128):
    """Function to generate a batch of data with batch_size number of data points
    Half of the data points will be Genuine-Genuine pairs and half will be Genuine-Forged pairs
    """
    orig_pairs = []
    forg_pairs = []

    for orig, forg in zip(orig_groups, forg_groups):
        orig_pairs.extend(list(itertools.combinations(orig, 2)))
        for i in range(len(forg)):
            forg_pairs.extend(
                list(itertools.product(orig[i : i + 1], random.sample(forg, 12)))
            )

    # Label for Genuine-Genuine pairs is 1
    # Label for Genuine-Forged pairs is 0
    gen_gen_labels = [1] * len(orig_pairs)
    gen_for_labels = [0] * len(forg_pairs)

    #     Here we create pairs of Genuine-Genuine image names and Genuine-Forged image names
    #     For every person we have 24 genuine signatures, hence we have
    #     24 choose 2 = 276 Genuine-Genuine image pairs for one person.
    #     To make Genuine-Forged pairs, we pair every Genuine signature of a person
    #     with 12 randomly sampled Forged signatures of the same person.
    #     Thus we make 24 * 12 = 288 Genuine-Forged image pairs for one person.
    #     In all we have 55 person's data in the training data.
    #     Total no. of Genuine-Genuine pairs = 55 * 276 = 15180
    #     Total number of Genuine-Forged pairs = 55 * 288 = 15840
    #     Total no. of data points = 31020

    # Concatenate all the pairs together along with their labels and shuffle them
    all_pairs = orig_pairs + forg_pairs
    all_labels = gen_gen_labels + gen_for_labels
    del orig_pairs, forg_pairs, gen_gen_labels, gen_for_labels
    all_pairs, all_labels = shuffle(all_pairs, all_labels)

    # Note the lists above contain only the image names and
    # actual images are loaded and yielded below in batches
    # Below we prepare a batch of data points and yield the batch
    # In each batch we load "batch_size" number of image pairs
    # These images are then removed from the original set so that
    # they are not added again in the next batch.

    k = 0
    pairs = [np.zeros((batch_size, cons.IMG_H, cons.IMG_W, 1)) for i in range(2)]
    targets = np.zeros((batch_size,))
    for ix, pair in enumerate(all_pairs):
        img1 = cv2.imread(pair[0], 0)
        img2 = cv2.imread(pair[1], 0)
        img1 = cv2.resize(img1, (cons.IMG_W, cons.IMG_H))
        img2 = cv2.resize(img2, (cons.IMG_W, cons.IMG_H))
        img1 = np.array(img1, dtype=np.float64)
        img2 = np.array(img2, dtype=np.float64)
        img1 /= 255
        img2 /= 255
        img1 = img1[..., np.newaxis]
        img2 = img2[..., np.newaxis]
        pairs[0][k, :, :, :] = img1
        pairs[1][k, :, :, :] = img2
        targets[k] = all_labels[ix]
        k += 1
        if k == batch_size:
            yield pairs, targets
            k = 0
            pairs = [
                np.zeros((batch_size, cons.IMG_H, cons.IMG_W, 1)) for i in range(2)
            ]
            targets = np.zeros((batch_size,))
