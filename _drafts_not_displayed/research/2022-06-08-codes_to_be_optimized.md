---
title: 深度学习：待优化的代码
date: 2022-06-08
categories: [科研]
tags: [技术]
img_path: /assets/img/
math: true
---

本文整理一下平时在写深度学习代码过程中遇到的细节问题，通常是找不到合适 API，直接使用笨方法实现的代码，列举于此，以期以后找到合适的优化方案。

## 1. Tensor 列表合并为 Tensor 

在对数据集操作时，经常需要将一个 Tensor 的列表转换为高一维的 Tensor。常见场景是从 Dataset 中取出数据子集，构造新的 Dataset 或 batch 等，例如构造 Split MNIST、取一小部分数据可视化（Tensorboard 的 `add_images` API 需要 Tensor 类型的 batch）。

设有一个 Tensor 列表，是从一个 Dataset 取出的数据：

```python 
tensor_list = [dataset[i][0] for i in range(100)]
```

如果直接用强制类型转换，则会报错：ValueError: only one element tensors can be converted to Python scalars。

```python
big_tensor = torch.tensor(tensor_list)
```

查了一下，目前的解决方案是（参考：<https://stackoverflow.com/questions/52074153/cannot-convert-list-to-array-valueerror-only-one-element-tensors-can-be-conver>）

```python
big_tensor = torch.tensor([tensor.detach().numpy() for tensor in tensor_list])
```

相当于先转换为 Numpy 的 Ndarray 的列表，再由它构造。但是这样会警告效率很低：UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 

