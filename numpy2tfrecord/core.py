"""Numpy to tfrecord converter

Author: Ryo Yonetani
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def convert_int_and_float(val):
    if isinstance(val, int):
        val = np.array(val, np.int64)
    if isinstance(val, float):
        val = np.array(val, np.float32)
    return val


def get_dtypes(sample: dict) -> dict:
    return {key: convert_int_and_float(sample[key]).dtype for key in sample.keys()}


def get_shapes(sample: dict) -> dict:
    return {key: convert_int_and_float(sample[key]).shape for key in sample.keys()}


class Numpy2Tfrecord:
    """
    Convert a collection of numpy data to tfrecord

    ```python
    import numpy as np
    from numpy2tfrecord import Numpy2Tfrecord

    converter = Numpy2Tfrecord()
    x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
    y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
    a = 5  # int
    b = 0.3  # float
    sample = {"x": x, "y": y, "a": a, "b": b}
    converter.add(sample)  # add data sample
    ...

    converter.export_to_tfrecord("test.tfrecord")  # export to tfrecord
    ```

    """

    def __init__(self):
        self.data = []
        self.dtypes = None
        self.shapes = None

    def add_sample(self, sample: dict) -> None:
        """
        Add a new sample to the list

        Args:
            sample (dict): new dataset sample to be parsed to tf.train.Example
        """
        if self.dtypes == None:
            self.dtypes = get_dtypes(sample)
        if self.shapes == None:
            self.shapes = get_shapes(sample)

        try:
            assert sample.keys() == self.dtypes.keys()
        except:
            raise (AssertionError("sample.keys() should be consistent"))

        try:
            assert self.dtypes == get_dtypes(sample)
        except:
            raise (AssertionError("dtype should be consistent"))

        try:
            assert self.shapes == get_shapes(sample)
        except:
            raise (AssertionError("shape should be consistent"))

        feature = dict()
        for key in sample.keys():
            element = convert_int_and_float(sample[key])

            if self.dtypes[key] == np.float32:
                feature[key] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=element.flatten())
                )
            elif self.dtypes[key] == np.int64:
                feature[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=element.flatten())
                )
            else:
                raise NotImplementedError(
                    f"dtype {self.dtypes[key]} is not currently supported"
                )
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.data.append(example)

    def add_list(self, samples: list[dict]) -> None:
        """
        Add a list of samples to the list

        Args:
            samples (list[dict]): list of samples, where each sample is a dict with the same keys and values with the same data type and shape.
        """
        for sample in samples:
            self.add_sample(sample)

    def add_batch(self, samples: dict) -> None:
        """
        Add a batch of samples to the list

        Args:
            samples (dict): a sample dictionary where the 0-th axis of all values corresponds to the batch size.
        """
        batch_size = next(iter(samples.values())).shape[0]
        for b in range(batch_size):
            sample = {key: samples[key][b] for key in samples.keys()}
            self.add_sample(sample)

    def export_to_tfrecord(self, filename: str) -> None:
        """
        Export added data stored in self.data to tfrecord.

        Args:
            filename (str): filename of tfrecord

        Note:
            With this function, self.dypes and self.shapes are saved as filename.info.
            This info file is necessary to reconstruct the original data type and shapes
            of each sample by a dataset created by `build_dataset_from_tfrecord`.
        """
        try:
            assert len(self.data) > 0
        except:
            print("No entries are found")
            return

        with tf.io.TFRecordWriter(filename) as writer:
            for example in tqdm(self.data):
                writer.write(example.SerializeToString())
        pickle.dump([self.dtypes, self.shapes], open(f"{filename}.info", "wb"))


def build_dataset_from_tfrecord(filename: str) -> tf.data.Dataset:
    """
    Build tf.data.Dataset from tfrecord file and .info file.

    Args:
        filename (str): filename of tfrecord

    Returns:
        tf.data.Dataset: Dataset of the data contained in the tfrecord
    """

    info_filename = f"{filename}.info"
    try:
        assert os.path.exists(info_filename)
    except:
        FileNotFoundError
    [dtypes, shapes] = pickle.load(open(info_filename, "rb"))

    def parse_tfrecord(example: tf.train.Example):
        feature_desc = dict()
        for key in dtypes.keys():
            if dtypes[key] == np.float32:
                feature_desc[key] = tf.io.VarLenFeature(tf.float32)
            elif dtypes[key] == np.int64:
                feature_desc[key] = tf.io.VarLenFeature(tf.int64)
            else:
                raise NotImplementedError(
                    f"dtype {dtypes[key]} is not currently supported"
                )
        features = tf.io.parse_single_example(example, feature_desc)

        sample = dict()
        for key in shapes.keys():
            sample[key] = tf.reshape(tf.sparse.to_dense(features[key]), shapes[key])
        return sample

    raw_dataset = tf.data.TFRecordDataset(filename)
    return raw_dataset.map(parse_tfrecord)
