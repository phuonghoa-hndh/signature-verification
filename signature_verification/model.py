import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2

import signature_verification.constants as cons


def euclidean_distance(vects):
    # Compute Euclidean Distance between two vectors
    x, y = vects
    sum_square = k.sum(k.square(x - y), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_square, k.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    return k.mean(
        y_true * k.square(y_pred)
        + (1 - y_true) * k.square(k.maximum(margin - y_pred, 0))
    )


def create_base_network_signet(input_shape):
    seq = Sequential()

    # Convolutional Layer 1
    seq.add(
        Conv2D(
            96,
            (11, 11),
            activation="relu",
            name="conv1_1",
            strides=(4, 4),
            input_shape=input_shape,
            kernel_initializer="glorot_uniform",
            data_format="channels_last",
        )
    )
    seq.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-06))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2), data_format="channels_last"))

    # Convolutional Layer 2
    seq.add(
        Conv2D(
            256,
            (5, 5),
            activation="relu",
            name="conv2_1",
            strides=(1, 1),
            kernel_initializer="glorot_uniform",
            data_format="channels_last",
        )
    )
    seq.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-06))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))
    seq.add(ZeroPadding2D((1, 1), data_format="channels_last"))

    # Convolutional Layer 3
    seq.add(
        Conv2D(
            384,
            (3, 3),
            activation="relu",
            name="conv3_1",
            strides=(1, 1),
            kernel_initializer="glorot_uniform",
            data_format="channels_last",
        )
    )
    seq.add(ZeroPadding2D((1, 1), data_format="channels_last"))

    # Convolutional Layer 4
    seq.add(
        Conv2D(
            256,
            (3, 3),
            activation="relu",
            name="conv3_2",
            strides=(1, 1),
            kernel_initializer="glorot_uniform",
            data_format="channels_last",
        )
    )
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))

    # Flatten and Fully Connected Layers
    seq.add(Flatten(name="flatten"))
    seq.add(
        Dense(
            1024,
            activation="relu",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=l2(0.0005),
        )
    )
    seq.add(Dropout(0.5))
    seq.add(
        Dense(
            128,
            activation="relu",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=l2(0.0005),
        )
    )

    return seq


def get_model():
    # Network definition
    input_shape = (cons.IMG_H, cons.IMG_W, 1)

    # Network definition
    base_network = create_base_network_signet(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Compute the Euclidean distance between the two vectors in the latent space
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b]
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True,
    )

    model = Model(inputs=[input_a, input_b], outputs=distance)

    model.compile(
        loss=contrastive_loss,
        optimizer=RMSprop(
            learning_rate=lr_schedule, rho=0.9, epsilon=1e-08, clipvalue=0.5
        ),
        metrics=["accuracy"],
    )

    return model
