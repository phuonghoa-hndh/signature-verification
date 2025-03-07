import os

import signature_verification.constants as cons
from signature_verification.split_data import split_data


def create_foler(tmp_path):
    for i in range(cons.NUM_WRITER):  # There are 55 writers -> 55 folders
        folder_name = f"{i}"
        os.mkdir(os.path.join(tmp_path, folder_name))


def test_split_data(tmp_path):
    # Test_size for train-test set is 0.4, train-val is 0.2
    len_expected_test = round(cons.NUM_WRITER * 0.4)
    len_temp_train = cons.NUM_WRITER - len_expected_test
    len_expected_val = round(len_temp_train * 0.2)
    len_expected_train = len_temp_train - len_expected_val

    path = tmp_path
    create_foler(path)
    train, val, test = split_data(path)

    assert len(train) == len_expected_train
    assert len(val) == len_expected_val
    assert len(test) == len_expected_test
