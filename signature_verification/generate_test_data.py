import itertools
import random

from sklearn.utils import shuffle

from signature_verification.prepare_data import get_groups


def generate_test_data(orig_groups, forg_groups):
    """
    This function is used to combine and shuffle 2 groups data and return their pairs and labels.
    Genuine-Genuine pairs is 1
    Genuine-Forged pairs is 0

    Then we can use the output to generate CSV file which contains 3 columns:
        img1(original signatures), img2(compared signatures), label(1 or 0)

    Args:
        orig_groups(list): the original signatures
        forg_groups(list): the forgery signatures

    Returns:
        all_pairs(list): the generated test data include orig and forg signatures
        all_labels(list): the generated test labels, 1 if they are from the same writer,  if they are not

    """
    orig_pairs = []
    forg_pairs = []

    for orig, forg in zip(orig_groups, forg_groups):
        orig_pairs.extend(list(itertools.combinations(orig, 2)))
        for i in range(len(forg)):
            forg_pairs.extend(
                list(itertools.product(orig[i : i + 1], random.sample(forg, 12)))
            )
            # 12 randomly sampled Forged signatures of the same person

    gen_ori_labels = [1] * len(orig_pairs)
    gen_for_labels = [0] * len(forg_pairs)

    all_pairs = orig_pairs + forg_pairs
    all_labels = gen_ori_labels + gen_for_labels
    del orig_pairs, forg_pairs, gen_ori_labels, gen_for_labels
    all_pairs, all_labels = shuffle(all_pairs, all_labels)

    return all_pairs, all_labels


if __name__ == "__main__":  # pragma: no cover
    # Get all the orig and forg signatures from get_groups
    orig_groups, forg_groups = get_groups(
        "./CEDAR", "./CEDAR_metadata/test_dir_list.txt"
    )
    all_pairs, all_labels = generate_test_data(orig_groups, forg_groups)

    max_size = 1000  # The desired test_data lines

    # Create CSV file
    with open("./CEDAR_test.csv", "a") as f:
        for idx, ((img1, img2), label) in enumerate(zip(all_pairs, all_labels)):
            if idx < max_size:
                img1 = img1.strip()
                img2 = img2.strip()
                label = str(label).strip()
                f.write(f"{img1},{img2},{label}\n")
