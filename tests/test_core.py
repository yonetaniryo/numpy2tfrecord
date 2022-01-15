import os

import numpy as np
import pytest


def test_add_export_build():
    from numpy2tfrecord.core import Numpy2Tfrecord

    converter = Numpy2Tfrecord()
    x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
    y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
    a = 5  # int
    b = 0.3  # float
    entry = {"x": x, "y": y, "a": a, "b": b}
    converter.add(entry)
    converter.add(entry)

    converter.export_to_tfrecord("test.tfrecord")
    dataset = converter.build_dataset_from_tfrecord("test.tfrecord")
    entry_reconstructed = next(dataset.as_numpy_iterator())
    for key in entry.keys():
        assert np.allclose(entry[key], entry_reconstructed[key])

    os.remove("test.tfrecord")
    os.remove("test.tfrecord.info")
