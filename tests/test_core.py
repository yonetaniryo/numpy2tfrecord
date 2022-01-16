import os

import numpy as np
import pytest


def test_add_sample():
    from numpy2tfrecord import Numpy2Tfrecord, build_dataset_from_tfrecord

    converter = Numpy2Tfrecord()
    x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
    y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
    a = 5  # int
    b = 0.3  # float
    sample = {"x": x, "y": y, "a": a, "b": b}
    converter.add_sample(sample)

    converter.export_to_tfrecord("test.tfrecord")

    dataset = build_dataset_from_tfrecord("test.tfrecord")
    sample_reconstructed = next(dataset.as_numpy_iterator())
    for key in sample.keys():
        assert np.allclose(sample[key], sample_reconstructed[key])

    os.remove("test.tfrecord")
    os.remove("test.tfrecord.info")


def test_add_list():
    from numpy2tfrecord import Numpy2Tfrecord, build_dataset_from_tfrecord

    converter = Numpy2Tfrecord()
    samples = [
        {
            "x": np.random.rand(64).astype(np.float32),
            "y": np.random.randint(0, 10),
        }
        for _ in range(32)
    ]
    converter.add_list(samples)

    converter.export_to_tfrecord("test.tfrecord")

    dataset = build_dataset_from_tfrecord("test.tfrecord")
    sample_reconstructed = next(dataset.as_numpy_iterator())
    for key in samples[0].keys():
        assert np.allclose(samples[0][key], sample_reconstructed[key])

    os.remove("test.tfrecord")
    os.remove("test.tfrecord.info")


def test_add_batch():
    from numpy2tfrecord import Numpy2Tfrecord, build_dataset_from_tfrecord

    converter = Numpy2Tfrecord()
    samples = {
        "x": np.random.rand(32, 64).astype(np.float32),
        "y": np.random.randint(0, 10, size=32).astype(np.int64),
    }
    converter.add_batch(samples)

    converter.export_to_tfrecord("test.tfrecord")

    dataset = build_dataset_from_tfrecord("test.tfrecord")
    sample_reconstructed = next(dataset.as_numpy_iterator())
    for key in samples.keys():
        assert np.allclose(samples[key][0], sample_reconstructed[key])

    os.remove("test.tfrecord")
    os.remove("test.tfrecord.info")
