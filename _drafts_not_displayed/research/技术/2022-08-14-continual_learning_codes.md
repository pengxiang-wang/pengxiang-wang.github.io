---
title: 持续学习代码复现
date: 2022-08-14
categories: [有趣的事情]
tags: [技术]
img_path: /assets/img/
---

本文给一个我写的持续学习的代码框架，并复现一些基本的持续学习模型。我将突出写持续学习代码时可能遇到的问题，突出思想性的东西。水平很菜，仅供参考。前置知识请参考[持续学习基础知识]() 与[《动手学深度学习》读书笔记]()。

# 主函数

参数：

- dataset 决定任务场景；
- approach 决定使用的整体方法，YAML，其中组合不同的优化器、损失函数、记忆等部件。包括了
- backbone 决定网络结构及其超参数；
- hyp


# 构造持续学习场景

构造持续学习场景的代码可以单独放在一起，整合成一个 CL 包。

## 构造数据集

持续学习的数据集都是其他标准数据集现构造出来的，需要代码把数据集划分成不同的任务。以 MNIST 为例，要构造 Permuted MNIST 和 Split MNIST 两种数据集，分别对应 TIL、CIL 场景。我定义了 PermutedMNIST 和 SplitMNIST 两个类，它们均采用与 PyTorch 提供的 MNIST 类相似的实例化方式（但不是直接继承），并加入额外的参数：
- PermutedMNIST：1. seeds：随机的 Permutation 种子；2. task_num，任务个数；
- SplitMNIST：class_split：任务的类别划分方法，以嵌套列表表示。
这些数据集类的特殊之处在于提供对 task 的索引，也可以方便地得到任务数量。

## 构造多头模型框架

[多头模型](https://pengxiang-wang.github.io/posts/continual_learning/#baseline多头模型)是持续学习场景必须的要求，对 TIL、CIL 都适用。它有各任务共用的特征提取器，每次遇到新出现的类别时会在特征提取器后构造一个输出头（一般是线性的）。我定义了一个 MultiHeadClassifier 类来实现，它可以完成增加类输出头、取部分输出头构成子模型等功能，而且也是 nn.Module 类，可与普通模型一样正常前向与反向传播。

在一开始实现时，我没有注意

并不总是按照0,1,2,3 的顺序来，如果出现别的顺序，由于 CrossEntropyLoss 的输出要求，只能转换。在训练时求一个临时label，而不是直接把label 修改成。

标签建议在 target

```python



```


# 持续学习算法



（微调模型）


HAT 的 mask 只能在模型结构中定义，因为每个 module 情况不一样，难以以统一的方法遍历神经元



dataloader 会在batch 和数据中间加一维以分隔；

多久采样一次 loss？一个epoch 一采？还是隔几个 batch？如何计算 loss？每次都算一次太占用。不科学。累加比较合适。

测试阶段，对每个任务 用 sub classifier 还是 fine-tuning classifier？

CIL 如何画三角图？


# iCaRL 方法



# 正则项


# 结构



# 可视化接口

TensorBoard

在 dataset.py 中定义了 show_in_tensorboard 函数，用于查看每个任务的图像。
r'r





想法：应当从 mask 原论文直接入手加与相似度有关的，而不是像 CAT 那样遭一个新的 KTA。

mask 初始化的问题，不能让第一个任务占了太多。

后向迁移与灾难性遗忘是一回事吗


前向迁移比后向迁移还要厉害嘛？


知识蒸馏技术与CL的关系


主动遗忘与模型容量分配问题？