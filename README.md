# numpy2tfrecord

Simple helper library to convert a collection of numpy data to tfrecord, and build a tensorflow dataset from the tfrecord.

## Installation
```sh
$ git clone git@github.com:yonetaniryo/numpy2tfrecord.git
$ cd numpy2tfrecord
$ pip install .
```

## Test
```sh
$ pytest -v 
```

## How to use
### Convert a collection of numpy data to tfrecord

You can add samples to a dataset by representing them in the form of a `dict`, and save them as a tfrecord.
```python
import numpy as np
from numpy2tfrecord import Numpy2Tfrecord

converter = Numpy2Tfrecord()
x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
a = 5  # int
b = 0.3  # float
sample = {"x": x, "y": y, "a": a, "b": b}
converter.add_sample(sample)  # add data sample
...

converter.export_to_tfrecord("test.tfrecord")  # export to tfrecord
```

You can also add a `list` of samples at once using `add_list`.
```python
samples = [
    {
        "x": np.random.rand(64).astype(np.float32),
        "y": np.random.randint(0, 10),
    }
    for _ in range(32)
]  # list of 32 samples

converter.add_list(samples)
```

Or add a batch of samples at once using `add_batch`.
```python
samples = {
    "x": np.random.rand(32, 64).astype(np.float32),
    "y": np.random.randint(0, 10, size=32).astype(np.int64),
}  # batch of 32 samples

converter.add_batch(samples)
```



### Build a tensorflow dataset from tfrecord
You can buld `tf.data.Dataset` from the exported tfrecord and info files.
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
