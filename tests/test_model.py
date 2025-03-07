import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, Sequential

from signature_verification.model import (
    contrastive_loss,
    create_base_network_signet,
    eucl_dist_output_shape,
    euclidean_distance,
    get_model,
)


def test_euclidean_distance():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected_output = np.array([[5.65685425], [5.65685425]])

    result = euclidean_distance([x, y])

    np.testing.assert_almost_equal(result, expected_output, decimal=6)


def test_eucl_dist_output_shape():
    shape1 = (100, 100)
    shape2 = (100, 100)
    expected_output_shape = (100, 1)

    result = eucl_dist_output_shape([shape1, shape2])
    assert result == expected_output_shape


def test_contrastive_loss():
    y_true = np.array([[1], [0]])
    y_pred = np.array([[0.5], [1.5]])
    expected_loss = (1 * 0.25 + (1 - 0) * 0) / 2

    result = contrastive_loss(y_true, y_pred)

    np.testing.assert_almost_equal(result, expected_loss, decimal=6)


def test_create_base_network_signet():
    fake_input_shape = (100, 100, 1)
    base_network = create_base_network_signet(fake_input_shape)

    # Check the input shape
    assert base_network.input_shape == (None, 100, 100, 1)

    # Check if the model is a Sequential model
    assert isinstance(base_network, Sequential)

    # Check the number of layers
    assert base_network.layers.__len__() == 18

    # Check the output shape
    output_shape = base_network.output_shape
    expected_output_shape = (None, 128)
    assert output_shape == expected_output_shape


def test_get_model():
    model = get_model()

    # Check if the model is a Keras Model
    assert isinstance(model, Model)

    # Check the input shapes
    assert model.input[0].shape == (None, 155, 220, 1)
    assert model.input[1].shape == (None, 155, 220, 1)

    # Check the number of layers
    assert len(model.layers) == 4

    # Check the final layer (Lambda)
    assert isinstance(model.layers[-1], Lambda)
    assert model.layers[-1].function == euclidean_distance

    # Check the output shape:
    # expected_output: <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'lambda')>
    assert model.output.shape == (None, 1)
