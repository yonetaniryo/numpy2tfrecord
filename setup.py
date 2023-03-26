from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="numpy2tfrecord",
    version="0.0.3",
    description="Convert a collection of numpy data to tfrecord",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/yonetaniryo/numpy2tfrecord",
    keywords="numpy, tfrecord, tensorflow",
    author="Ryo Yonetani",
    author_email="yonetani.vision@gmail.com",
    license="MIT License",
    packages=["numpy2tfrecord"],
    install_requires=[
        "numpy>=1.22.0",
        "tensorflow;platform_system=='Windows'",
        "tensorflow;platform_system=='linux'",
        "tensorflow-macos;platform_system=='darwin'",
    ],
)
