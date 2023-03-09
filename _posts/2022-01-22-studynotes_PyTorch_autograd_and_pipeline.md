---
title: PyTorch 学习笔记（一）：自动微分，简单模型的实现
date: 2022-01-22
categories: [科研]
tags: [读书笔记, 《动手学深度学习》, 机器学习, 技术]
img_path: /assets/img/
math: true
---

本系列博文是我学习深度学习框架的学习笔记。深度学习框架大同小异，只须学习一种的原理，其他的都可以快速上手。我使用的是 PyTorch。笔记将着重强调代码原理、思想的理解，而不是具体的代码。

- PyTorch 官方文档：<https://pytorch.org/docs/stable/index.html>
- PyTorch 中文文档：<https://pytorch-cn.readthedocs.io/zh/latest/>

本系列博文的参考书为亚马逊团队编写的 [Dive into Deep Learning (PyTorch 版)](https://d2l.ai)，编排顺序基本遵从此书。导师最近很推荐这本书。这是一本把深度学习从头开始讲的技术书，虽然大部分内容是会的，但难得找到一本书在代码上讲得系统，看一遍也是很有好处的。我计划是利用寒假时间看一看，整理出一套笔记。B 站上有[李沐](https://space.bilibili.com/1567748478/) 主讲的[配套课程](https://c.d2l.ai/zh-v2/)可供参考。

本文介绍深度学习框架的基本数据结构——Tensor 及其核心功能——自动微分，并搭建几个最简单的监督学习模型，主要参考书的：
- 2.5 节：自动微分；
- 3.2-3.3 节：线性回归的从零开始实现、简洁实现；
- 3.5-3.7 节：Softmax 多分类的从零开始实现、简洁实现；
- 4.1-4.3 节：多层感知机（MLP）的从零开始实现、简洁实现。

------------------------------

在开始前还是提示一下如何安装 PyTorch。去[官网](https://pytorch.org)翻到 Install PyTorch，根据自己机器的系统等信息选择后，用下面生成的指令安装。如果不想用或没有 GPU，选择 CPU 版本即可；如果想用，请参考 [PyTorch 学习笔记：使用 GPU]()，了解 CUDA 的意思后选择合适的 CUDA 版本安装。
![选择](PyTorch_installation.png)
安装过程如果报错，尝试使用国内镜像，参见 [Conda 学习笔记](https://pengxiang-wang.github.io/posts/studynotes_conda/)。


# 基本数据结构：Tensor

PyTorch 是深度学习框架，预备知识一定是基本的数据结构、数据操作。**张量**（Tensor）是 PyTorch 的基本数据结构，它的性质和用法就是数学上的张量，在[这篇博文]()已详细讲述。书中 2.1,2.3,2.4 等节大部分篇幅在讲述 Tensor 的基本用法，这些与 Numpy 也是一致的，就跳过了。 

这篇博文也总结过，PyTorch 和 Numpy 的基本数据结构本质都是数学上的张量，而且 PyTorch 是基于 Numpy 的，为什么还要自己封装一个 Tensor 类型？书中第 2 章开头总结的不错，PyTorch 在 Tensor 中融入了深度学习相关的功能：
- 在 GPU 上加速计算（Numpy 只能在 CPU）；
- 储存梯度、计算图等信息，实现自动微分功能。


## 自动微分

[**自动微分**](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)（求导）是深度学习框架的主要功能，顾名思义就是给出一个函数后，即可直接算出在某点的导数（梯度）值（注意，并不是待求导函数的表达式）。计算图、链式法则是自动微分基于的原理，但也不需要搞明白其底层实现方式，只要会用即可。

> 自动微分只能完成数值计算，而不是fu只能求在某点的导数值，而不能求出导函数的表示

需要理解的是，自动微分功能是实现在 Tensor 里的，自动微分的计算过程和结果都是存的 Tensor 的属性中的：
- `grad_fn`：存放待求导函数（的计算图）；
- `grad`：存放求得的导数向量（Tensor）。
这个 Tensor 即为被求导点。


假设要求 $$\frac{\operatorname{d} y}{\operatorname{d} \mathbf{x}}_{\mathbf{x}=\mathbf{x}_{0}}$$，以求 $$y = 2\mathbf{x}^T \mathbf{x}$$ 在 $$x_0 = (0,1,2,3)^T$$ 点的梯度为例，完整的自动微分过程如下：

1. 定义 $$x_0$$：将 $$x_0$$ 点的值以 tensor 的形式赋给变量 `x`
```python
    x = torch.arange(4.0)
```
2. 开启求导模式：把 tensor `x` 的 `requires_grad` 属性设为 True
```python
    x.requires_grad_(True)
```  
> 求导模式可以在 Tensor 构造时即刻开启，只需在构造的函数传入参数 `requires_grad=True`。例如上面 1,2 两步可合为 `x = torch.arange(4.0, requires_grad=True)`。
{: .prompt-tip }
3. 定义被求导函数 $$y$$：将含 `x` 的 torch 表达式赋给变量 `y` （此时 tensor `y` 存放了计算图）
```python
    y = 2 * torch.dot(x, x)
```
4. 求导：调用 `y` 的 `backward` 方法，导数值存放在 `x` 的 `grad` 属性中（与 `x` 维数相同）
```python
    y.backward()
```

注意点：
- 存放求导结果的 `grad` 属性是累加的：第一次求导前默认为 0，求导后将结果叠加到 0 上，第二次求导后会叠加到第一次的结果上。所以如需反复求导一定要**清零**。清零的方法：
  ```python
    x.grad.zero_()
  ```
- 被求导函数可以额外打包成一个 Python 函数赋给 `y`（只要函数里面用的都是 torch 的表达式）；
- 构建计算图极容易粗心，一定注意好求导模式的开关，不在不该的地方引入计算图。除了修改 `requires_grad_` 属性，还可以：
    - 全局地关闭求导模式，用以下代码包裹：
    ```python
        with torch.no_grad():
    ```
    - 分离：即去掉 `grad_fn` 存放的计算图，只保留 tensor 值。以下代码将 `y` 分离成 `u`：
    ```python
        u = y.detach()
    ```
- 上面求导要求 $$y$$ 必须是标量，而 $$x$$ 可以是向量。事实上 $$y$$ 也可以是向量，需要在 backward 函数中加参数，见下例。

> 使用自动微分工具可以画一个函数导数的图像，参见书第 4 章画各种激活函数及其导数。例，Sigmoid 函数：
> ```python
> import matplotlib.pyplot as plt
> x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
> y = torch.sigmoid(x)
> y.backward(torch.ones_like(x))
> plt.plot(x.detach(), x.grad)
> ```
> 注意此时 y 是向量，要令其每一维对 x 每一维分别求导，须在 backward 函数中加 `torch.ones_like(x)` 参数。
{: .prompt-tip }


# 深度学习模型的 Pipeline

深度学习的完整流程如下：
1. 数据预处理；
2. 定义模型、损失函数、优化器、初始化等；
3. 训练模型；
4. 测试模型。
以下各节具体讲解细节。

在 PyTorch 实现中，以上每一步都包含许多值得单独讲的专题。作为学习笔记系列的第一篇，先从简单模型出发，将这些流程实现一遍，好对 PyTorch 有个整体的认识。将介绍三个简单模型，分别是：
- 线性回归；
- Softmax 多分类；
- 多层感知机（MLP）。
每个模型都分从头开始实现和简洁实现两种实现方法。简洁实现是调用 PyTorch 提供的高级 API，从头开始实现是自己写训练过程等细节，仅利用 PyTorch 的自动微分功能。这样有利于理解深度学习框架相比于其他包为深度学习带来的极大方便。



## 一、线性回归

本节欲训练线性回归模型：$$ \mathbf{y} = \mathbf{X}\mathbf{w} + b + \epsilon $$。

PyTorch 作深度学习使用的数据集都是它定义的 Dataset 类型。这里用到的数据暂时不涉及该类型，而是手动生成的普通的 Tensor。本例生成方法：给定 $$\mathbf{w}, b$$ 的真实值，按正态分布（`torch.normal`）生成 $$\mathbf{X}, \mathbf{y}$$，用一个 `synthetic_data(w, b, num_examples)` 函数实现（可以自己写一下试试，练练 Tensor 的使用）。

### 从头开始实现

深度学习通常是按批（batch）训练的，因此数据 $$\mathbf{X}, \mathbf{y}$$ 还需按一定的批数据量（batch_size）划分成各批数据。代码没有简单地切片成 batch 并存到列表里，而是通过**生成器**（generator）生成（参考我的 [Python 笔记]()），这样的好处是每次训练需要时调用一次生成器，它现场给你生成新的一个 batch 的数据（**是这些数据拼接成的矩阵**），而无需一开始就划好，否则一开始切片 batch 这种预处理工作就要花费很多时间，会使训练过程迟迟不能开动。生成器函数 `data_iter(batch_size, features, labels)` 也可以自己试试，要注意两个细节：shuffle 的实现只需 shuffle 索引；如何 num_examples 不能整除 batch_size，尾部如何处理。

先看训练框架：
```python
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```
中间的三句分别是前向传播、后向传播、梯度下降。每次从生成器生成一个 batch 的数据训练用。最外层循环为轮数，每一轮结束都要统计一下当前训得的模型在整个训练集上的 loss。
> 统计的时候不需求导引入计算图，可以用 `with torch.no_grad():` 包裹，纯粹为了减少计算量，不包裹也不会出错（例如下面的简洁实现就没有包裹）。但是有的地方引入计算图会引起混乱，如下面 `sgd()` 函数里的梯度下降更新式，一定要包裹。
{: .prompt-warning }

训练过程就是不断更新参数 `w,b`，中间三句是如何更新的呢？答案就是自动微分。在此代码前应对 `w,b` 作初始化：
```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```
开启求导模式后，定义被求导函数，在深度学习中就是损失函数 $$l = \sum_{i=1}^{batch_size} loss(net(X, y, b))$$。`net()` 是模型函数（输出预测值），`loss()` 是损失函数，它们都是事先定义的 Python 函数（参见“自动微分”的注意点 2）：
```python
def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

net = linreg
loss = squared_loss
```
> 要注意的是 $$X$$ 不是一个数据点，而是一批数据拼成的矩阵，在写上述函数时要注意应当完成对一整个 batch 的数据的操作。
{: .prompt-warning }
被求导函数就是通过 `l = loss(net(X, w, b), y)` 和 `l.sum()` 定义的。注意前者得到的 l 是一个长度为 batch_size 的向量，因为 `X,y` 是一个 batch 的数据，它们是一起计算的（矩阵化比 for 循环要快），求和后才是这一 batch 的损失函数。

接下来 `.backward()` 执行求导。由于只有 `w,b` 开启了求导模式，也只会求 $$\frac{\partial l}{\partial w}, \frac{\partial l}{\partial b}$$，导数结果存放在 `w,b` 的 grad 属性中。

下一步是梯度下降，打包成一个函数 `sgd(params, lr, batch_size)`。首先一个小细节是捆起来传参数列表 `params`，除了代码易于维护，另外就是将其变为可变类型，直接修改 `w,b` 而不需返回。再说一遍，无需单独传导数，已经存放在 grad 属性中。
```python
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```
主体部分是每个参数都执行梯度下降，`lr` 是事先定义好的学习率，还要除以 batch_size 是因为之前 l 的计算没有平均，放到这里，也起到规范化步长的作用。注意梯度下降的表达式会构造新的计算图，导致混乱，一定要以 `with torch.no_grad():` 包裹。

梯度下降结束后，`w,b` 的值随即更新。由于下一个 batch 还要求导，不要忘了给 grad 清零。这里把它巧妙地写在 `sgd()` 函数里面，能充分利用 for 循环遍历参数，而不需分别写 `w,b`。

训练结束后，我们得到 `w,b`，就得到了训练好的模型，调用 `net(x, w, b)` 可对 x 进行预测。本文比较了训练的 `w,b` 和生成数据时真实的 `w,b` 的误差。

### 简洁实现

上述实现中很多步骤可以换成 PyTorch 简洁的 API 实现。

首先是现成的生成器，PyTorch 里有现成的 Dataloader 类可使用（[文档](https://pytorch.org/docs/stable/data.html)）。这个类的实例就是生成器，构造函数为
```python
from torch.utils import data
data_iter = data.Dataloader(dataset, batch_size, shuffle=True)
```
此句为从 dataset 构造大小为 batch_size 的数据生成器。dataset 是 PyTorch 的 Dataset 类型（`torch.utils.data.Dataset`），需要按规则构造，当然也有现成的数据集（从 `torchvision.datasets` 里 import 即可）。构造规则是一个比较麻烦的事，将在别的笔记中再讨论。书中这里的代码暂时省略了对它的讨论。

其他选项：
- `shuffle`：指定需不需要打乱数据的顺序；
- `sampler`,`batch_sampler`：自定义采集 batch 的方式，传入的是 `torch.utils.data.Sampler` 类型。不指定则采用顺序采集（`shuffle=False`）或随机采集（`shuffle=True`）。

第二是现成的模型、损失函数。上述线性模型函数 `linreg`、平方损失 `squared_loss` 无需自己定义，在 PyTorch 中有现成的：
```python
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
loss = nn.MSELoss()
```
PyTorch 中的线性模型就是 `nn.Linear(m, n)`，`m, n` 分别为输入、输出神经元数。`nn.Sequential()` 将不同的 Layer 串联起来构造成一个大的模型，它其实是一个容器，通过下标索引 `net[0]` 可以选中各层。它们的类型是 PyTorch 自己的 `torch.nn.modules` 里的“模块”类型，各个“模块”具有树状的父子关系，例如本例是父模块 `net`（nn.Sequential）下嵌套子模块 `net[0]`（nn.Linear）。在[此笔记]()中将介绍复杂的深度网络，将见到更多复杂的模块组合。

这些现成的函数作用相当于“自动微分”注意点 2 的所说的函数，但是还是有不一样的地方。它们的一个重要特点是**模型参数都存放到这里面了**，它是真正意义上的模型。通过以下代码，体会一下如何查看模型参数：
```python
# 参数初始化
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 打印参数
print(f'{net[0].weight.data:f} {net[0].bias.data:f}')
```

优化器也是事先构造好的：
```python
import torch
trainer = torch.optim.SGD(net.parameters(), lr=0.01)
```
即实例化一个 `Optimizer` 优化器类，优化器可从 `torch.optim` 里挑选，有 SGD, Adam 等。注意这里 `net.parameters()`，模型参数从一开始就与优化器绑定到一起了。（注意这个事情，有助于理解下面 `trainer` 不需要传模型参数。）

训练步骤简化为：
```python
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```
到现在看，简洁性体现在以下几个方面，PyTorch 的设计者把所有有门槛的、需要深度理解的细节全都隐藏了：
- 不需要自己写模型、损失等函数，省去了考虑那些复杂的矩阵操作；
- 优化器不需要自己写，而且 `trainer` 什么参数也不用传（甚至模型参数），调用一下 `step()` 搞定；甚至 grad 清零的工作挪到了 `trainer.zero_grad()`，同样不需要传模型参数。

还有几个小细节也足以体现：
- `net()`不需要显式地传入模型参数，直接写 `net(X)` 即可；
- `l` 不需要 `sum()` 了，直接 `backward()` 后面也能知道什么意思；

一开始学习深度学习框架，只需看懂高级 API 表面的工作流程，会写即可，并不特别需要了解这些 API 背后的细节。

## 二、Softmax 多分类

第二个模型是 Softmax 多分类模型：$$\mathbf{O} = \mathbf{X} \mathbf{W} + \mathbf{b}, \mathbf{y} = Softmax(\mathbf{O})$$。

此部分做的是图像分类问题，用的是 Fashion-MNIST 图像数据集，做 10 分类。这是会涉及使用 Dataset 类型的使用，但仅限于调用 PyTorch 自带的 Dataset 数据集实例。在安装 PyTorch 时可以看到，它包含 3 个库，torch 即深度学习框架，是工具；而 torchvision，torchaudio 是专门提供例子的库：数据集，网络，变换，分成视觉和语音两部分。因此 PyTorch 提供的图像数据集在 torchvision.datasets 里。这里面常见的 MNIST、CIFAR、ImageNet 数据集都有，见[官方文档](https://pytorch.org/vision/stable/datasets.html)。

读取数据集即从其中的类中创建实例：
```python
import torchvision
from torchvision import transforms

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
```
它将从 `root` 指示的目录（路径格式见[Linux 学习笔记]()）中寻找 FashionMNIST 数据，如果没有：`download=False` 时报错，`download=True` 时将从网上下载数据到该目录内，同时语句返回 Dataset 类型的变量 `mnist_train`,`mnist_test`，它包含一对对 $$(X,y)$$ 元组，可以像列表一样中括号索引。要注意，需要规定 `transform=transform.ToTensor()`，这样里面的 X 才是 Tensor 类型，否则默认为 PIL 类型（Python Image Library，是 Python 图像处理标准库 Pillow 表示图像的类型）。


### 从头开始实现

对于数据生成器，这里直接使用简洁实现——Dataloader，没有再从头实现。要注意测试数据也需要构造 Dataloader。
```python
train_iter = data.Dataloader(mnist_train, batch_size, shuffle=True)
test_iter = data.Dataloader(mnist_test, batch_size, shuffle=True)
```

以下是从头开始实现定义的模型和损失函数，这里不打算细讲，看看就好，基本上是各种矩阵操作、广播机制的巧妙运用。可以看到，自己写这些东西是比较麻烦的，就是因为需要注意一整个 batch 的数据传入的问题，这就涉及更高阶的矩阵操作。
```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True) # keepdim 是为了下面用广播机制
    return X_exp / partition

def softmax_linreg(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

net = softmax_linreg
loss = cross_entropy
```
注意，X 是 (28,28) 图像，将其拉直这一操作放到了模型里（`X.reshape(-1, 28*28)`），而不是数据预处理过程中。其他小细节是没有把参数 `W,b` 传入函数参数，而是当作全局变量了（其实这样不太好）。`W,b` 也像之前一样手动构造并初始化：
```python
W = torch.normal(0, 0.01, size=(28*28, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
```

书中到这里第一次涉及测试过程的写法。测试过程涉及准确率，在从头实现中也是要自己写的。注意它和 loss 一样要考虑一整个 batch 传入的问题，if 语句就是在检查是否为单个数据的：
```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat,argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)
```
书中用了自己构造的数据结构 `Accumulator` 作统计工作，有点麻烦，我翻译了一下伪代码：
```python
with torch.no_grad():
    for X, y in test_iter:
        # 计算累加 loss(net(X), y)
        # 计算累加 accuracy(net(X), y)
    # 打印统计后的 loss 和 accuracy
```

从此模型开始，作者自己写了一个 `Animator` 类用于展示每个 epoch 训练情况，可实时画出训练 loss，测试准确率等。这个东西实在没必要自己写，有现成的工具 TensorBoard 很好用，参考我的 [TensorBoard 学习笔记]()。

### 简洁实现

简洁实现仍然使用了现成的模型、损失函数、优化器。优化器的简化同上，这里就看一看模型和损失函数的定义：
```python
from torch import nn

net = nn.Sequential(nn.Flatten(), nn.Linear(784,10))
loss = nn.CrossEntropyLoss()
```
等等！Softmax 函数哪儿去了？这是一个重要的问题。实际上，Softmax 函数放到了 `CrossEntropyLoss` 里面了，也就是说 `net(X)` 输出的是未经 Softmax 规范化的预测。PyTorch 这样设计的原因涉及背后的计算机理，是为效率服务的，详见书 3.7.2 节。

这里的初始化用了另一套 API：
```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```
对于 `nn.Module` 模块，它的 `apply` 方法可以将传入的函数递归地作用到它包含的所有模块上。本例即将 `init_weights` 作用在 `net`（`nn.Sequential`），`net[0]`（`nn.Flatten`），`net[1]`（`nn.Linear`）。由于 `init_weights` 里的 if 语句，只对 `net[1]` 应用 `nn.init.normal_`。`nn.init` 的用法详见[笔记（三）]()。

另外，PyTorch 里没有实用的求准确率的 API，因为实在是没必要，自己写两个小函数就解决了。

测试过程同上从头开始实现。


## 三、多层感知机（MLP）

本节问题仍为图像多分类问题。MLP 模型与上面 Softmax 多分类相比，无非是网络层数由一层变为多层，层间引入了激活函数。

### 从头开始实现

模型定义和初始化大同小异。这里值得关注的新东西是：参数打包成 `nn.Parameter` 实例。前面见过简洁实现中模型的参数 `net.parameters()` 就是 `nn.Parameter` 类型的，它是进一步封装的类。而这里即使没有用到简洁实现的 `nn.modules`，也能当作一般的 Tensor 正常使用，还是很灵活的（原因：`nn.Parameter` 源代码定义了 `__new__()` 方法，它返回 Tensor 类型）。另外，下面的 `@` 运算符是 PyTorch 重载的，等价于矩阵乘法 `torch.matmul()`。
```python
W1 = nn.Parameter(torch.randn(784, 256, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(256, requires_grad=True))
W2 = nn.Parameter(torch.randn(256, 10, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(10, requires_grad=True))
params = [W1, b1, W2, b2]

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(X, a)

def mlp(X):
    X = X.reshape((-1, num_outputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)

net = mlp
loss = nn.CrossEntropyLoss()
```

### 简洁实现

这里唯一的变化是定义模型多了两个模块：隐藏层和激活函数。不再详述。
```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```


<br>

总结一下，有了自动微分这一工具后，深度学习看似简单，但是上面所有的从头开始实现，写起来真的特别麻烦，要顾虑很多细节如矩阵化，有很多坑。深度学习框架的高级 API 不仅写法简单，写模型就跟搭积木一样，不用考虑细节，而且采取了额外的预防措施确保数值稳定性，帮助编程人员避免从头实现可能遇到的陷阱。所以以后如非学习目的，能用框架就不要自己手写！