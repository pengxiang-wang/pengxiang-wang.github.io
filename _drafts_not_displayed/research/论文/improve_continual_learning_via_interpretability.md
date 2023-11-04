---
title: 科研笔记：可解释性与持续学习
date: 2022-10-08
categories: [科研]
tags: [论文笔记, 持续学习]
img_path: /assets/img/
math: true
---


## 继续持续学习的研究

### AdaHAT


### 工程性应用（快速发C会）


持续学习在一个具体的别人都不知道的任务上，自己就是baseline，只要能自圆其说它的意义
 寻求具体项目的问题
李健昊提到的 HAT

博鸿博的 Twitter 数据场景



## 持续学习的可解释性（XCL）

### 探索性工作

用已有的解释性方法找一找持续学习模型里的规律，看看有什么启发

- Mask + Network Dissection，先从人工的概念出发分析，找规律看看，重合程度的神经元保留了什么信息？

找一找相关的论文，看看有没有分析具体某个模型的可解释性的就能发出来的


用 Captum 探索持续学习网络的各个部分，是Attribution，local的
Aggregate Sample-based results 来得到 global的

- 在测试数据统计一些量，
  - 按类别，按任务，按照epoch
- 
  - 平均
    - 不平均，按照某个batch
    - 按照某种阈值（prob > 0.99）统计，
    - 按照label统计
    - 按照influence score加权
    - 不能用全部，或者随机，尤其是数据不均衡时。对图像更是没用。
  - 按类别平均，task 和 class 的表格

统计feature的，layer的，neuron的
- mask 重合的神经元在不同子网络里有什么作用，在不同任务中有什么作用
  - 找一个查看神经元
- 通过importance来剪枝
  - https://www.youtube.com/watch?v=hY_XzglTkak
  - 比一般的任意剪枝能提升效果，主要在召回率？方面Neuron importance based pruning can help us to increase TP, F1 and Recall scores and reduce FN
  - 
- 
captum 可以 influential 











### 引入更有可解释性的机制

寻找 ICICLE 类似的工作（Active Interpretability）



Ma Yi 的工作，可解释性？

工具 NTK

### 提出持续学习中现象特有的解释方法


提出一些可解释方法。

把可解释性套在持续学习上，需要突出持续学习的特色，如何突出？

- 解释为什么一个样本在训练完下一个任务之后 ，预测结果发生了改变
  - 论文里如何写意义：对于本来分对了的样本后来分错，就是遗忘，有了这样的解释可以帮助诊断（这样写在论文里足够吗？）
  - 方法1：用influential sample 来解释？对旧任务的样本，如果前后分错，比较各自训练集中最influential的influence得分，计算出来某种指标。
  - 方法2：通过引入可解释性机制（model-specific）



## 可解释性辅助持续学习

### 压缩模型

用可解释性里的特征重要性方法，压缩持续学习模型

### 应用解释信息

当做下一个任务的训练信息
。
用解释性生成的实例选出样本点
influence functions 增强样本

Saliency map mix-up，




## 其他

### 梯度操控法：优化器

和朱容宇发TAG相关的工作（二作），可以学习一下写论文的方法

传送（Symmetry Teleportation for Accelerated Optimization）应用到持续学习