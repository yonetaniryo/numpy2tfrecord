import os

import numpy as np


def test_convert_sample():
    from numpy2tfrecord import Numpy2TFRecordConverter, build_dataset_from_tfrecord

    with Numpy2TFRecordConverter("test.tfrecord") as converter:
        x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
        y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
        a = 5  # int
        b = 0.3  # float
        sample = {"x": x, "y": y, "a": a, "b": b}
        converter.convert_sample(sample)

    dataset = build_dataset_from_tfrecord("test.tfrecord")
    sample_reconstructed = next(dataset.as_numpy_iterator())
    for key in sample.keys():
        assert np.allclose(sample[key], sample_reconstructed[key])

    os.remove("test.tfrecord")
    os.remove("test.tfrecord.info")


def test_convert_list():
    from numpy2tfrecord import Numpy2TFRecordConverter, build_dataset_from_tfrecord

    with Numpy2TFRecordConverter("test.tfrecord") as converter:
        samples = [
            {
                "x": np.random.rand(64).astype(np.float32),
                "y": np.random.randint(0, 10),
            }
            for _ in range(32)
        ]
        converter.convert_list(samples)

    dataset = build_dataset_from_tfrecord("test.tfrecord")
    sample_reconstructed = next(dataset.as_numpy_iterator())
    for key in samples[0].keys():
        assert np.allclose(samples[0][key], sample_reconstructed[key])

    os.remove("test.tfrecord")
    os.remove("test.tfrecord.info")


def test_convert_batch():
    from numpy2tfrecord import Numpy2TFRecordConverter, build_dataset_from_tfrecord

    with Numpy2TFRecordConverter("test.tfrecord") as converter:
        samples = {
            "x": np.random.rand(32, 64).astype(np.float32),
            "y": np.random.randint(0, 10, size=32).astype(np.int64),
        }
        converter.convert_batch(samples)

    dataset = build_dataset_from_tfrecord("test.tfrecord")
    sample_reconstructed = next(dataset.as_numpy_iterator())
    for key in samples.keys():
        assert np.allclose(samples[key][0], sample_reconstructed[key])

    os.remove("test.tfrecord")
    os.remove("test.tfrecord.info")
