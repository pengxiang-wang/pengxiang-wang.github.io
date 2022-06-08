---
title: 读书笔记：《动手学深度学习》第 2,3,4 章：预备知识，简单网络的实现
date: 2022-01-22
categories: [科研]
tags: [读书笔记, 《动手学深度学习》, 技术]
img_path: /assets/img/
math: true
---

导师最近很推荐这本书。这是一本把深度学习从头开始讲的技术书，虽然大部分内容是会的，难得找到一本书在代码上讲得系统，看一遍也是很有好处的。我计划是利用寒假时间看一看，整理出一套笔记。

此笔记内容遵从书的编排方式，但内容不局限于书中内容，我可能会补充很多东西，包括自己的理解。这篇笔记着重强调代码原理的理解，具体代码我放在另一篇笔记：[PyTorch 速查手册]()上。


## 书籍信息 

### [Dive into Deep Learning (PyTorch 版)](https://d2l.ai)
- 作者：亚马逊团队


------------------------------


第 1 章前言部分又把机器学习、深度学习的基础知识讲了一遍，直接跳过。第 2 章介绍 PyTorch 的预备知识，包括张量的基本操作、自动微分等，第 3、4 章开始搭建简单网络。

## 张量

张量（tensor）是 [PyTorch](https://pytorch.org/docs/stable/tensors.html) 内置的数据结构，用来表示数学上的张量（0阶：数；1阶：向量；2阶：矩阵，……）。其他框架也有类似的数据结构，如 [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/Tensor) 中的 tensor，[Numpy](https://numpy.org/doc/stable/reference/arrays.ndarray.html) 中的 ndarray。

Python 表示数据最常用的就是 Numpy 向量了，为什么深度学习框架还要再用一套 tensor？主要是多了深度学习相关的功能：例如在 GPU 上加速计算（Numpy 只能在 CPU），自动微分的功能。

除了这些功能，其他对数学上张量的操作它们都有相应的 API。这些东西实在没必要像书中从头到尾过一遍，因为数学向量、矩阵有哪些操作大家都很熟悉，现查就可以了。下面只谈新功能的原理，那些与 Numpy 的 ndarray 相同的概念与功能请移步我的 [Numpy 学习笔记]()。



### 兼容性

我所谓兼容性是指与其他 Python 对象互相转换的方便程度。




### 自动微分

[自动微分](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)（求导）是深度学习框架的主要功能，顾名思义就是给出一个函数后，即可直接算出在某点的导数（梯度）值（注意，并不是求导函数的表达式）。

可为什么在张量这里介绍呢？张量不是函数呀！那是因为张量中有个属性（`grad_fn`)是用来存函数（计算图）的。计算图、链式法则是自动微分基于的原理，但也不需要搞明白其底层实现方式，只要会用即可。

假设要求 $$\left.\frac{\operatorname{d} y}{\operatorname{d} \mathbf{x}}\right|_{\mathbf{x}=\mathbf{x}_{0}}
$$，以求 $$y  = 2\mathbf{x}^T \mathbf{x}$$ 在 $$x_0 = (0,1,2,3)^T$$ 点的梯度为例，完整的自动微分过程如下：

1. 定义 $$x_0$$：将 $$x_0$$ 点的值以 tensor 的形式赋给变量 `x`
```
    x = torch.arange(4.0)
```
2. 开启求导模式：把 tensor `x` 的 `requires_grad_` 属性设为 True
```   
    x.requires_grad_(True)
```  
3. 定义被求导函数 $$y$$：将含 `x` 的 torch 表达式赋给变量 `y` （此时 tensor `y` 存放了计算图）
```
    y = 2 * torch.dot(x, x)
```
4. 求导：调用 `y` 的 `backward` 方法，导数值存放在 `x` 的 `grad` 属性中（与 `x` 维数相同）
```
    y.backward()
```

注意点：
- y 必须是标量，而 x 可以是向量；
- 存放求导结果的 `grad` 属性是累加的：第一次求导前默认为0，求导后将结果叠加到 0 上，第二次求导后会叠加到第一次的结果上。所以如需反复求导一定要**清零**；
- 被求导函数可以额外打包成一个 Python 函数赋给 `y`，只要函数里面用的都是 torch 的表达式；
- 构建计算图极容易粗心，一定注意好求导模式的开关，不在不该的地方引入计算图。除了修改 `requires_grad_` 属性，还可以：
    - 全局地关闭求导模式，用以下代码包裹：
    ```
        with torch.no_grad():
    ```
    - 分离变量：即去掉 `grad_fn` 存放的计算图，只保留 tensor 值。以下代码将 `y` 分离成 `u`：
    ```
        u = y.detach()
    ```   


## 深度学习模型的 Pipeline


本书在第 3、4 章介绍了三个简单网络，每个模型都分从头开始实现和简单实现两部分，有点啰嗦。





以下是几个简单网络的实现方式。








