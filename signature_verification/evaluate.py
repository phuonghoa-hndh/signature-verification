import argparse

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import signature_verification.constants as cons
from signature_verification.model import get_model
from signature_verification.prepare_data import generate_batch, get_groups


def process_test_metadata(path):
    data = pd.read_csv(path, header=None)
    pairs_data = list(zip(data.iloc[:, 0], data.iloc[:, 1]))
    labels_list = data.iloc[:, 2].tolist()

    return pairs_data, labels_list


pairs_data, labels_list = process_test_metadata(cons.TEST_CSV_PATH)


def gen_data(batch_size=1):
    k = 0
    pairs = [np.zeros((batch_size, cons.IMG_H, cons.IMG_W, 1)) for i in range(2)]
    targets = np.zeros((batch_size,))
    for ix, pair in enumerate(pairs_data):
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
        targets[k] = labels_list[ix]
        k += 1
        if k == batch_size:
            yield pairs, targets
            k = 0
            pairs = [
                np.zeros((batch_size, cons.IMG_H, cons.IMG_W, 1)) for i in range(2)
            ]
            targets = np.zeros((batch_size,))


def compute_accuracy_roc(predictions, labels):
    """
    What: This function is to compute ROC accuracy with a range of thresholds on distances.
    Why: Get the accuracy of data file and get the best threshold on distances between each pair of that data file
    When: We already have the labels of its data file and the model can predict each pairs of that file
    How:
        predictions = []
        labels = []
        for i in range(data_file):
            (img1, img2), label = next(data_file)
            predictions.append(model.predict(img1, img2))
            labels.append(label)
        compute_accuracy_roc(predictions, labels)
    Args:
        predictions: list of predictions
        labels: list of labels
    Returns:
        max_acc(float): maximum accuracy score
        threshold(float): This threshold is to decide which pair is forged or genuine.
    """
    # The min and max distance between 2 images
    d_max = np.max(predictions)
    d_min = np.min(predictions)
    # The number of labels equal 1: n_same, equal 0: n_diff
    n_same = np.sum(labels == 1)
    n_diff = np.sum(labels == 0)

    step = 0.01
    max_acc = 0
    best_thresh = -1

    for d in np.arange(d_min, d_max + step, step):
        idx1 = predictions <= d
        idx2 = predictions > d

        tpr = float(np.sum(labels[idx1] == 1)) / n_same
        tnr = float(np.sum(labels[idx2] == 0)) / n_diff
        acc = 0.5 * (tpr + tnr)

        if acc > max_acc:
            max_acc, best_thresh = acc, d

    return max_acc, best_thresh


def get_val_threshold(model):
    """
    This function is to get threshold to comparing 2 images of validation set.
    Args:
        model: the model which needs to be evaluated.
    Returns:
        threshold(float): the threshold of validation set

    """
    orig_val, forg_val = get_groups(
        "CEDAR",
        cons.VAL_PATH,
    )
    val_gen = generate_batch(orig_val, forg_val, 1)
    pred, true_y = [], []
    for i in range(1000):
        (img1, img2), label = next(val_gen)
        true_y.append(label)
        pred.append(model.predict([img1, img2])[0][0])

    max_acc, threshold = compute_accuracy_roc(np.array(pred), np.array(true_y))
    return threshold


def predict_test_data(model):
    test_gen = gen_data()

    threshold = get_val_threshold(model)

    true_y, predictions = [], []
    for i in range(1000):
        (img1, img2), label = next(test_gen)
        true_y.append(label)

        predict = model.predict([img1, img2])[0][0]
        if predict > threshold:
            predictions.append(0)
        else:
            predictions.append(1)

    return true_y, predictions


def main(args):
    model = get_model()
    model.load_weights(args.weights)

    true_y, predictions = predict_test_data(model)
    threshold = get_val_threshold(model)

    print("Threshold: ", threshold)
    print(
        classification_report(
            true_y, predictions, labels=[1, 0], target_names=["is_real", "is_forged"]
        )
    )


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )
    args = parser.parse_args()
    main(args)
