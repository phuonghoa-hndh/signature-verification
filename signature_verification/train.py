import argparse

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import signature_verification.constants as cons
from signature_verification.model import get_model
from signature_verification.prepare_data import generate_batch, get_groups


def main(batch_size, num_epoch):
    # Train model
    orig_train, forg_train = get_groups("CEDAR", "./CEDAR_metadata/train_dir_list.txt")
    orig_val, forg_val = get_groups("CEDAR", "./CEDAR_metadata/val_dir_list.txt")
    model = get_model()

    callbacks = [
        EarlyStopping(monitor="train_loss", patience=3, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(
            "./signet-cedar-{epoch:03d}.weights.h5",
            verbose=1,
            save_weights_only=True,
        ),
    ]

    history = model.fit(
        generate_batch(orig_train, forg_train, batch_size),
        steps_per_epoch=cons.NUM_TRAIN_SAMPLES // batch_size - 1,
        epochs=num_epoch,
        validation_data=generate_batch(orig_val, forg_val, batch_size - 1),
        validation_steps=cons.NUM_VAL_SAMPLES // batch_size,
        callbacks=callbacks,
    )

    return history


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        "-bs",
        action="store",
        type=int,
        default=128,
        help="The mini batch size. Default: 128",
    )
    parser.add_argument(
        "--num_epoch",
        "-e",
        action="store",
        type=int,
        default=5,
        help="The maximum number of iterations. Default: 5",
    )

    args = parser.parse_args()
    main(**vars(args))
