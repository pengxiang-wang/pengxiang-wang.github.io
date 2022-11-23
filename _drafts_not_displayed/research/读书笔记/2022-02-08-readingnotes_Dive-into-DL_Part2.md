---
title: 读书笔记：《动手学深度学习》Part 2：深度学习训练的问题及其解决方案
date: 2022-02-08
categories: [科研]
tags: [读书笔记, 《动手学深度学习》, 技术]
img_path: /assets/img/
math: true
---


## 书籍信息 

### [Dive into Deep Learning (PyTorch 版)](https://d2l.ai)
- 作者：亚马逊团队
- 配套课程：[李沐](https://space.bilibili.com/1567748478/) 主讲，视频上传于 B 站。链接：<https://c.d2l.ai/zh-v2/>
- 本 Part 内容：第 4 章中间部分，介绍深度学习训练时可能出现的问题及其解决方案：过拟合/欠拟合、数值稳定性。
- 
------------------------------
本部分讨论深度学习训练的问题，在书中对应的章节讲解理论特多，代码实现较少，我将书中理论的东西整合到[这篇笔记]，此笔记仍主要关心实践，本笔记分别对应了这篇笔记的各章节，给出了代码实现。我会略过理论部分的讲解，并补充书中未涉及的代码实现。


按照实际搭建一整套流程。



在真正开始训练前，作double check 。cs231n





# 一、过拟合、欠拟合

此图是可以实时画出来的，使用作者的 `Animator` 类或 Tensorboard 等工具。

### 权重衰减（L2 正则化）

#### 从头开始实现

只需在前向传播 loss 中加入正则项：
```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#...
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y) + lambd * l2_penalty(w)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
        #...
```
一个小细节：`lambda` 是 Python 的关键字，不能占用，此超参数改写为 `lambd`。

#### 简洁实现

在 PyTorch 的高级 API，权重衰减功能在优化器中提供：
```python
trainer = torch.optim.SGD(
    [{'params':net[0].weight, 'weight_decay':3}, {'params':net[0].bias}],
    lr=lr)
```
相当于线性回归简洁实现中修改了优化器的 `params` 参数，现在仔细研究一下这个参数。[PyTorch 中文文档](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/)上写：

> Optimizer 也支持为每个参数单独设置选项。若想这么做，不要直接传入 Variable 的 iterable，而是传入 dict 的 iterable。

此时属于后一种情况。传入各参数的字典必须包含一个参数键值：`'params':参数`（其中“参数”为 `nn.Parameters` 类），其余的键值负责该参数的其他选项，如本例中的键 `'weight_decay':wd`，其值 `wd` 代表正则化系数 $$\lambda$$。当然，Optimizer 自己的构造参数（写在字典外面）能全局设置，例如本例 `lr=lr` 指定了整个学习率。如果与字典中的设置冲突，以字典为准。本例没有全局设置 `weight_decay`，因为不想给 bias 正则化。

### 暂退法（Dropout）

#### 从头开始实现

#### 简洁实现


# 二、数值稳定性

本节讨论数值稳定性问题，首先考察深度学习训练的主要算法——反向传播算法。


简单的**随机初始化**就能很好地避免的上面的问题。常用做法是使用正态分布，通常均值（取0）、方差是固定的。之前的例子都是我们手动初始化了的，对简洁实现的高级 API 来说，即使不手动初始化，框架也将默认使用随机初始化，说明此方法非常常用且普遍。

PyTorch 也提供了几十种初始化方法，[下一 Part]() 会系统讲解初始化的代码。每种层都有自己的默认初始化方法。

## 解决数值稳定性问题的方法


# 三、环境和分布偏移
以线性回归为例，**L2 正则化** 就是在损失函数中加正则项：

$$ L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n SquaredLoss (\mathbf{w}^T \mathbf{x}^{(i)} + b, y^{(i)}) + \frac{1}{\lambda} \left\| \mathbf{w}\right\|^2$$
书 4.7 节讲解了加正则化的网络训练时的前向传播计算图长什么样、反向传播的公式推导，我直接跳过。


# 超参数优化





自动优化。
