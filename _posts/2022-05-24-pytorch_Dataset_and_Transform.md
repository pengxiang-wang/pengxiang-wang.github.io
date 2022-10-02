---
title: PyTorch 理解自定义数据集与预处理
date: 2022-05-24
categories: [科研]
tags: [技术]
img_path: /assets/img/
---

本文整理一下 PyTorch 如何自定义数据集。关于此话题，参考官方 Tutorial，知乎[这篇文章](https://zhuanlan.zhihu.com/p/130673468)也讲得不错，言简意赅。

PyTorch API 中的数据集都是定义了一个 `torch.utils.data.Dataset` 的子类，数据集都是这个子类的实例。必须这样做，因为后面构造 Dataloader 只接收 Dataset 类型，而整个训练过程都是对 Dataloader 的操作。

之前[《动手学深度学习》读书笔记](https://pengxiang-wang.github.io/posts/readingnotes_Dive-into-DL_Part1/)中用的一律是现成的数据集，即 PyTorch 预定义了的一些 Dataset 子类，定义在 `torchvision.datasets` 中。但在实际项目中，大多需要自定义数据集，例如持续学习里需要构造任务数据集。

# 自定义 Dataset

自定义数据集就需要自己写 Dataset 子类。通常一个具体的数据集对应一个该类。该类需要定义三个方法：

```python
class MyDataset(Dataset):
    def __init__(self, ...):
        ...

    def __len__(self):
        ...
        return len # 返回数据集数据个数

    def __getitem__(self, index):
        ...
        return image, label # 返回第 index 个数据 + 标签

```


在 [Python 笔记](https://pengxiang-wang.github.io/posts/studynotes_Python/) 中说过，`__getitem__()`,`__len__()` 属于特殊类方法，前者规定了 `len()` 函数作用在类实例上的返回值，后者规定了索引 `mydataset[index]` 的返回值。为什么必须实现 `__geiitem__()`,`__len__()` 呢？因为 Dataloader 的核心业务都是在调用这两个方法。不定义会报错。


不管你怎么存储数据集的数据（放在什么数据结构里），按什么顺序实现，只要把 `__getitem__()` 确实做了它该做的事情，就大功告成了。比如：

- 有人喜欢在 Dataset 构造时就把数据集全部读取出来（数据集本体存放在 `__init__()` 某个实例属性中），`__getitem__()` 直接索引即可；
- 还有的人喜欢在 `__init__()` 中只给本地文件路径，在调用 `__getitem__()` 时现场读取相应的数据。区别只是效率问题，逻辑上。

# 预处理 Transform

在 PyTorch 中，数据预处理通常在上述 Dataset 子类中的 `__getitem__()` 方法中定义，即索引到某数据时对其临时做预处理：

```python
class MyDataset(Dataset):
    def __init__(self, ..., transform, target_transform):
        ...
        self.transform = transform
        self.target_transform = target.transform
        ...
    def __getitem__(self, index):
        ...
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```
传入的 `transfrom` 和 `target_transform` 分别是对数据和标签的预处理变换，是可调用对象就行：可以是函数，输入处理前的（单个）数据，输出处理后的数据。在 `torchvision.transforms` 预定义了很多常用的变换，都是定义了 `__call__()` 方法的类。例如 `ToTensor()`，

```python
from . import functional as F

class ToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
```
最核心的一句话就是 `__call__()` 方法中的调用了 `F.to_tensor()` 这个变换。

我们也可以照葫芦画瓢构造自己的预处理变换。


# 例子

下面给出几个例子。

第一个例子是[官方 tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 里的示例：

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

这个 Dataset 子类可以从本地读取带标签的图像数据集。本地数据要求图像存放在一个 img_dir 目录下，另有一个标签数据文件 annotation_file，是 csv 文件，第一列为图像文件名，第二列为对应图像的标签。可以看到属于上面的第二种实现方式：在 `__getitem__()` 根据 annotation_file 的信息获得图像本地路径，现场读取图像。

第二个例子是 `torchvision.datasets` 里的数据集 MNIST，学习一下人家是怎么定义的。这些 Dataset 子类的一大特点是不仅可以读取本地的 MNIST 数据集，还能在本地文件存在时从指定网站上下载（通过 `download` 参数控制）。它属于第一种方式：在 `__init__()` 方法中事先将数据集本体存放在了实例属性 `self.data`,`self.targets` 中了，`__getitem__()` 直接索引这两个变量即可。注意读取进来的源格式是 PIL，通常机器学习需要转化为 Tensor，一般会传入 `transform=ToTensor()`。此外，这个代码加入了好多纠错机制，非常地健壮。

```python

class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
```
