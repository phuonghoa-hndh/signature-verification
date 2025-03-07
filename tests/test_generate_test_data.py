import pytest

from signature_verification.generate_test_data import generate_test_data


@pytest.fixture
def create_temp_data(tmpdir):
    orig_groups = [
        [f"{tmpdir}/orig_{i}_{j}.png" for j in range(1, 25)] for i in range(1, 23)
    ]

    forg_groups = [
        [f"{tmpdir}/forg_{i}_{j}.png" for j in range(1, 25)] for i in range(1, 23)
    ]

    for group in orig_groups + forg_groups:
        for img in group:
            with open(img, "w") as f:
                f.write("data")

    return orig_groups, forg_groups


def test_generate_test_data(create_temp_data):
    orig_groups, forg_groups = create_temp_data
    actual_all_pairs, actual_all_labels = generate_test_data(orig_groups, forg_groups)

    # There are 22 people in test set, 24 choose 2 = 276 Genuine-Genuine image pairs for one person.
    # 24 * 12  = 288 Genuine-Forged image pairs for one person.
    num_orig_pairs = 276 * 22
    num_forg_pairs = 288 * 22

    assert len(actual_all_pairs) == num_orig_pairs + num_forg_pairs
    assert len(actual_all_labels) == num_orig_pairs + num_forg_pairs

    for pair, label in zip(actual_all_labels, actual_all_pairs):
        if label == 1:
            # if both signatures from the same person
            assert any(pair[0] in group and pair[1] in group for group in orig_groups)
        elif label == 0:
            # if 1 is original and the other is forgery
            assert any(
                pair[0] in orig_group and pair[1] in forg_group
                for orig_group, forg_group in zip(orig_groups, forg_groups)
            )
