import os

from sklearn.model_selection import train_test_split

PATH = "./CEDAR"


def split_data(path):
    """
    Splits the data into training, validation, and test sets
    There are 55 writers, so using `train_test_split` with parameter `test_size=0.4`
    splits the data into 22 writers for testing and the remaining for training.
    Another split on the training set with `test_size=0.2` creates the validation set.
    Args:
        path: The path of the folder where the data is stored (here is CEDAR)
    Returns:
        A tuple: Containing the training, validation, and test lists.
    """
    # Get the list of all directories and sort them
    dir_list = next(os.walk(path))[1]
    dir_list.sort()

    train_dir_list, test_dir_list = train_test_split(
        dir_list, test_size=0.4, random_state=1
    )
    train_dir_list, val_dir_list = train_test_split(
        train_dir_list, test_size=0.2, random_state=1
    )

    return train_dir_list, val_dir_list, test_dir_list


if __name__ == "__main__":  # pragma: no cover
    # Get txt file of train, val, test list
    train_dir_list, val_dir_list, test_dir_list = split_data(PATH)

    with open("./train_dir_list.txt", "w") as f:
        f.write(",".join(train_dir_list))

    with open("./val_dir_list.txt", "w") as f:
        f.write(",".join(val_dir_list))

    with open("./test_dir_list.txt", "w") as f:
        f.write(",".join(test_dir_list))
