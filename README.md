# numpy2tfrecord

Simple helper library to convert numpy data to tfrecord and build a tensorflow dataset.

## Installation
```sh
$ git clone git@github.com:yonetaniryo/numpy2tfrecord.git
$ cd numpy2tfrecord
$ pip install .
```
or simply using pip:
```sh
$ pip install git+https://github.com/yonetaniryo/numpy2tfrecord
```


## How to use
### Convert a collection of numpy data to tfrecord

You can convert samples represented in the form of a `dict` to `tf.train.Example` and save them as a tfrecord.
```python
import numpy as np
from numpy2tfrecord import Numpy2TfrecordConverter

with Numpy2TfrecordConverter("test.tfrecord") as converter:
    x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
    y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
    a = 5  # int
    b = 0.3  # float
    sample = {"x": x, "y": y, "a": a, "b": b}
    converter.convert_sample(sample)  # convert data sample
```

You can also convert a `list` of samples at once using `convert_list`.
```python
with Numpy2TfrecordConverter("test.tfrecord") as converter:
    samples = [
        {
            "x": np.random.rand(64).astype(np.float32),
            "y": np.random.randint(0, 10),
        }
        for _ in range(32)
    ]  # list of 32 samples

    converter.convert_list(samples)
```

Or a batch of samples at once using `convert_batch`.
```python
with Numpy2TfrecordConverter("test.tfrecord") as converter:
    samples = {
        "x": np.random.rand(32, 64).astype(np.float32),
        "y": np.random.randint(0, 10, size=32).astype(np.int64),
    }  # batch of 32 samples

    converter.convert_batch(samples)
```

So what are the advantages of `Numpy2TfRecordConverter` compared to `tf.data.datset.from_tensor_slices`? 
Simply put, when using `tf.data.dataset.from_tensor_slices`, all the samples that will be converted to a dataset must be in memory. 
On the other hand, you can use `Numpy2TfRecordConverter` to sequentially add samples to the tfrecord without having to read all of them into memory beforehand..



### Build a tensorflow dataset from tfrecord
Samples once stored in the tfrecord can be streamed using `tf.data.TFRecordDataset`.

```python
from numpy2tfrecord import build_dataset_from_tfrecord

dataset = build_dataset_from_tfrecord("test.tfrecord")
```

The dataset can then be used directly in the for-loop of machine learning.

```python
for batch in dataset.as_numpy_iterator():
    x, y = batch.values()
    ...
```
