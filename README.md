# numpy2tfrecord

Simple helper library to convert numpy data to tfrecord, and build tensorflow dataset from the tfrecord.

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

## Getting Started
```python
import numpy as np
from numpy2tfrecord import Numpy2Tfrecord

converter = Numpy2Tfrecord()
x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
a = 5  # int
b = 0.3  # float
entry = {"x": x, "y": y, "a": a, "b": b}
converter.add(entry)  # add data entry
...

converter.export_to_tfrecord("test.tfrecord")  # export to tfrecord
dataset = converter.build_dataset_from_tfrecord("test.tfrecord")  # load tfrecord and build tf.data.Dataset
```