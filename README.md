# Signature Verification


## Description
The model used to verify the user's signature is Signet (as shown). This model takes 2 images as input and outputs the Euclidean distance between the 2 signatures. Each image goes through a common CNN backbone, which produces a 128-dimensional vector representing the signature image in vector space. If the Euclidean distance between the two vectors is small, the signatures match and are considered genuine. If the distance is large, the second signature is deemed to be fake.


## Train

This guide is to check if the package's performance is similar to the model's performance on a chosen dataset.

- Run 'train.py'

To train the model:
```shell
python train.py

```
