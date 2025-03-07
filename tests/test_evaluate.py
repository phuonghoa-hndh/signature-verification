from unittest.mock import patch

import numpy as np
import pandas as pd

import signature_verification.constants as cons
from signature_verification.evaluate import (
    compute_accuracy_roc,
    gen_data,
    process_test_metadata,
)


def test_process_test_metadata(tmpdir):
    # Create a fake test data csv
    path = tmpdir.join("fake_test_data.csv")

    fake_csv_data = pd.DataFrame(
        [
            "./CEDAR/38/original_38_13.png,./CEDAR/38/forgeries_38_20.png,0",
            "./CEDAR/12/original_12_22.png,./CEDAR/12/forgeries_12_14.png,0",
            "./CEDAR/33/original_33_10.png,./CEDAR/33/forgeries_33_24.png,0",
        ]
    )

    csv_data = fake_csv_data[0].str.split(",", expand=True)

    csv_data.to_csv(path, index=False, header=False)

    pairs_data, labels_list = process_test_metadata(path)

    expected_pairs_data = [
        ("./CEDAR/38/original_38_13.png", "./CEDAR/38/forgeries_38_20.png"),
        ("./CEDAR/12/original_12_22.png", "./CEDAR/12/forgeries_12_14.png"),
        ("./CEDAR/33/original_33_10.png", "./CEDAR/33/forgeries_33_24.png"),
    ]
    expected_labels_list = [0, 0, 0]

    assert pairs_data == expected_pairs_data
    assert labels_list == expected_labels_list


def test_compute_accuracy_roc():
    # Create a fake result of model predictions
    predictions = np.array(
        [0.89941, 0.234, 0.5543, 0.7543, 1.245, 2.54687, 0.441247, 1.1215, 2.3333]
    )

    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])

    max_acc, best_thresh = compute_accuracy_roc(predictions, labels)

    # Check the range of accuracy and threshold
    assert 0.0 <= max_acc <= 1.0
    assert 0.234 <= best_thresh <= 2.54687


def create_cv2_imread(filepath, flags):
    return np.ones((cons.IMG_H, cons.IMG_W), dtype=np.uint8) * 255


def test_gen_data():
    with patch("cv2.imread", side_effect=create_cv2_imread):
        batch_size = 2
        actual_gen = gen_data(batch_size=batch_size)
        pairs, targets = next(actual_gen)

        assert pairs[0].shape == (batch_size, cons.IMG_H, cons.IMG_W, 1)
        assert pairs[1].shape == (batch_size, cons.IMG_H, cons.IMG_W, 1)
        assert targets.shape == (batch_size,)
