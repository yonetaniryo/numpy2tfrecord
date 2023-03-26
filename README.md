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
$ pip install numpy2tfrecord
```


## How to use
### Convert a collection of numpy data to tfrecord

You can convert samples represented in the form of a `dict` to `tf.train.Example` and save them as a tfrecord.
```python
import numpy as np
from numpy2tfrecord import Numpy2TFRecordConverter

with Numpy2TFRecordConverter("test.tfrecord") as converter:
    x = np.arange(100).reshape(10, 10).astype(np.float32)  # float array
    y = np.arange(100).reshape(10, 10).astype(np.int64)  # int array
    a = 5  # int
    b = 0.3  # float
    sample = {"x": x, "y": y, "a": a, "b": b}
    converter.convert_sample(sample)  # convert data sample
```

You can also convert a `list` of samples at once using `convert_list`.
```python
with Numpy2TFRecordConverter("test.tfrecord") as converter:
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
with Numpy2TFRecordConverter("test.tfrecord") as converter:
    samples = {
        "x": np.random.rand(32, 64).astype(np.float32),
        "y": np.random.randint(0, 10, size=32).astype(np.int64),
    }  # batch of 32 samples

    converter.convert_batch(samples)
```

So what are the advantages of `Numpy2TFRecordConverter` compared to `tf.data.datset.from_tensor_slices`? 
Simply put, when using `tf.data.dataset.from_tensor_slices`, all the samples that will be converted to a dataset must be in memory. 
On the other hand, you can use `Numpy2TFRecordConverter` to sequentially add samples to the tfrecord without having to read all of them into memory beforehand..



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

### Speeding up PyTorch data loading with `numpy2tfrecord`!
https://gist.github.com/yonetaniryo/c1780e58b841f30150c45233d3fe6d01

```python
import os
import time

import numpy as np
from numpy2tfrecord import Numpy2TfrecordConverter, build_dataset_from_tfrecord
import torch
from torchvision import datasets, transforms

dataset = datasets.MNIST(".", download=True, transform=transforms.ToTensor())

# convert to tfrecord
with Numpy2TfrecordConverter("mnist.tfrecord") as converter:
    converter.convert_batch({"x": dataset.data.numpy().astype(np.int64), 
                        "y": dataset.targets.numpy().astype(np.int64)})

torch_loader = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=os.cpu_count())
tic = time.time()
for e in range(5):
    for batch in torch_loader:
        x, y = batch
elapsed = time.time() - tic
print(f"elapsed time with pytorch dataloader: {elapsed:0.2f} sec for 5 epochs")

tf_loader = build_dataset_from_tfrecord("mnist.tfrecord").batch(32).prefetch(1)
tic = time.time()
for e in range(5):
    for batch in tf_loader.as_numpy_iterator():
        x, y = batch.values()
elapsed = time.time() - tic
print(f"elapsed time with tf dataloader: {elapsed:0.2f} sec for 5 epochs")
```

⬇️

```
elapsed time with pytorch dataloader: 41.10 sec for 5 epochs
elapsed time with tf dataloader: 17.34 sec for 5 epochs
```
