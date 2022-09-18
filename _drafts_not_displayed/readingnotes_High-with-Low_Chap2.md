---
title: 读书笔记：《低维模型下的高维数据分析》第2、3章：稀疏信号恢复
date: 2022-03-08
categories: [科研]
tags: [读书笔记, 《低维模型下的高维数据分析》]
img_path: /assets/img/
math: true
---


## 书籍信息 



### [High-Dimensional Data Analysis with Low-Dimensional Models: Principles, Computation, and Applications](https://book-wright-ma.github.io)

- 作者：
    - [John Wright](http://www.columbia.edu/~jw2966/) - 哥伦比亚大学
    - [马毅](https://people.eecs.berkeley.edu/~yima/) - UC Berkeley


------------------------------

## 稀疏信号恢复问题

第2章、第3章都是围绕以下**向量重建问题**：

$$ \mathbf{A} \mathbf{x} \text{(unknown)} = \mathbf{y} \text{(observation)} $$

其中 $$\mathbf{y} \in \mathbb{R}^m, \mathbf{x} \in \mathbb{R}^n, \mathbf{A} \in \mathbb{R}^{m\times n}$$。此式应理解为：为了了解未知的向量 $$\mathbf{x}$$，使用不同的 $$m$$ 个向量（$$\mathbf{A}$$ 的各行向量）与之做内积，得到 $$m$$ 个观测值（$$\mathbb{y}$$ 的各分量）

重建问题的目标是：已知观测 $$\mathbf{y}$$ 与生成矩阵 $$\mathbf{A}$$，恢复出未知的 $$\mathbf{x}$$。

当观测数据不够，即 $$m < n$$ 时，这个重建问题是病态的（ill-posed），因为此方程解不唯一。为了使问题 well-posed，对 $$\mathbf{x}$$ 作**稀疏性假设**：$$\mathbf{x}$$ 非零分量数不能太多。向量重建问题变为**稀疏信号恢复问题**。

直观上看，对方程的解作了稀疏限定后，解集范围变小，有可能使不唯一解缩小为唯一解。事实是这样吗？稀疏到什么程度才能让解唯一？这是这两章的“恢复理论”要解决的问题。



以下是几个例子：

### 核磁共振

### 图像压缩

### 字典学习

### 带遮挡物的人脸识别

## 数学工具

### 向量稀疏性的衡量

0范数 1范数


### 矩阵线性无关性的衡量


## 恢复理论

### 从l0到l1


### 
