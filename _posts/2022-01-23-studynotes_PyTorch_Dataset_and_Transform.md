---
title: PyTorch 学习笔记（二）：自定义数据集，数据预处理
date: 2022-01-23
categories: [科研]
tags: [机器学习,《动手学深度学习》，技术]
img_path: /assets/img/
---

本文总结 PyTorch 中与数据集以及对它的预处理的知识。知乎的[这篇文章](https://zhuanlan.zhihu.com/p/130673468)讲得不错，言简意赅。也可参考[官方教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)。


------------------------------


PyTorch 中的数据集都是定义了一个 `torch.utils.data.Dataset` 类型，数据集都是这个类型的实例。必须这样做，因为后面构造 Dataloader 只接收 Dataset 类型，而整个训练过程都是对 Dataloader 的操作。我们已经在[笔记（一）]() 中学习了 Dataloader，所以本文专心于学习 Dataset。

PyTorch 预定义了很多数据集，将其打包成 Dataset 类型，定义在 `torchvision.datasets` 中，常用基本的 MNIST、CIFAR、ImageNet 等数据集都有，更多的见文档：<https://pytorch.org/vision/stable/datasets.html>。但在实际项目中，大多需要用自己定制的数据集，这时需要自定义 Dataset 类型（例如持续学习里需要构造任务数据集）。





# 通用 Dataset 模版

自定义数据集就需要自己写 Dataset 类。一个数据集对应一个该类的实例。最简单的自定义 Dataset 需要定义三个方法：

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

在 [Python 笔记](https://pengxiang-wang.github.io/posts/studynotes_Python/) 中说过，`__getitem__()`,`__len__()` 属于特殊类方法，前者规定了 `len()` 函数作用在类实例上的返回值，后者规定了索引 `mydataset[index]` 的返回值。要注意的是，除了构造函数 `__init__()`，`__geiitem__()`,`__len__()` 也是必须实现的，因为数据生成器 Dataloader 的核心业务就是在调用这两个方法，入股不定义会报错。

对 `__geiitem__()`,`__len__()` 的实现是灵活的，记住，不管你怎么存储数据集的数据（放在什么数据结构里），按什么顺序实现，只要把 `__getitem__()` 确实做了它该做的事情，就大功告成了。比如：

- 有人喜欢在 Dataset 构造时就把数据集全部读取出来（数据集本体存放在 `__init__()` 某个实例属性中），`__getitem__()` 直接索引即可；
- 还有的人喜欢在 `__init__()` 中只给本地文件路径，在调用 `__getitem__()` 时现场读取相应的数据。区别只是效率问题。


# 数据预处理变换

在 PyTorch 中，数据预处理都归结为设计**变换函数 `transform`**，形式上是一个 Python 函数，输入处理前的数据（一般是 Tensor），输出处理后的数据（保持维度不变）。

这个变换函数作用到数据的方式是：包裹在 Dataset 类里，并在 `__getitem__()` 方法中作用到数据上，即索引到某数据时对其临时做预处理：
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
传入的 `transfrom` 和 `target_transform` 分别是对数据和标签的预处理变换。

PyTorch 预定义了很多这种 Python 函数，完成预处理变换，称为函数型变换（functional transforms）：<https://pytorch.org/vision/stable/transforms.html#functional-transforms>。这里不展开讲解，因为更常用的实现方式是下述的可调用类。


数据预处理变换通常是有一些超参数的，例如旋转变换的角度，等等。上述这种传参的方式，如果 `transform` 是普通的 Python 函数，那么这些超参数将无法一并传入，所以数据预处理变换一般实现为**可调用类**，在类的构造函数中包裹超参数。下例来自[官方文档](https://pytorch.org/vision/stable/transforms.html)：
```python
class MyRotationTransform:

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
```

PyTorch 也预定义了很多实用的预处理变换类，定义在 `torchvision.transforms` 中，包括数据标准化、降维、数据增强等，在文档中有详细描述：<https://pytorch.org/vision/stable/transforms.html>。其中常用的值得学习一下：
- 数据类型转换：记住一个即可，很多预定义的数据集都是 PIL 类型的（Python Pillow 库定义的类型），无法直接用于训练或测试，`ToTensor()`将其转换为 Tensor 类型；
- 数据标准化：`Normalize()`；
- 数据增强：提供了对图像的各种增强方法，如缩放（`Resize()`）、裁剪（`CenterCrop()`、`FiveCrop()`、`RandomCrop()`）、旋转（`RandomRotation()`）等。

> Dataset 的这种构造方式似乎只能传一个变换，如果是自定义的可以把各种操作写在同一个复杂的变换里。对于上述预定义变换，如果要应用多个变换，PyTorch 也设计了一个 `Compose()` 变换，用于组合多个预定义的变换以方便传入 Dataset 的参数。
{: .prompt-tip }


# PyTorch 预定义的数据集与预处理变换

本节我们学习几个 PyTorch 预定义的例子，看看官方是怎么写 Dataset 和 transform 的，对我们自己写也会有所启发。

## 官方教程示例

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


## MNIST

第二个例子是 `torchvision.datasets` 里的数据集 MNIST。这些 Dataset 类的一大特点是不仅可以读取本地的 MNIST 数据集，还能在本地文件存在时从指定网站上下载（通过 `download` 参数控制）。它属于第一种方式：在 `__init__()` 方法中事先将数据集本体存放在了实例属性 `self.data`,`self.targets` 中了，`__getitem__()` 直接索引这两个变量即可。注意读取进来的源格式是 PIL，通常机器学习需要转化为 Tensor，一般会传入 `transform=ToTensor()`。此外，这个代码加入了好多纠错机制，非常地健壮。

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


## torchvision.transforms.Normalize()

第三个例子是经典的数据标准化 `torchvision.transforms.Normalize()`。可以看到，这个可调用类只是一个壳，真正的实现包裹在了 `F.normalize()` 这个函数中。我们在写自己的变换时也最好这样模块化，这是一个好习惯。

另外一个有趣的地方是，这个变换类继承自 `nn.Module`，也就是说，它也可以当做一个网络层，放在由 `nn.Module` 组织的网络结构里使用。这样的二用，巧妙地省去了很多代码。（这也是为什么需要提供均值、方差两个参数的原因，数据预处理一般默认是数据集的均值、方差，而在网络结构中，它们可能是要学习的参数。）

```python
class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    _assert_image_tensor(tensor)

    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor
```