---
title: PyTorch 学习笔记：卷积神经网络（CNN）
date: 2022-02-11
categories: [科研]
tags: [读书笔记, 《动手学深度学习》, 机器学习, 技术]
img_path: /assets/img/
math: true
---

本文介绍如何搭建卷积神经网络结构（前置知识为 [nn.Module 模块的使用]()），并介绍常见 CNN 结构的搭建方法和简单调用方式。本文参考 [Dive into Deep Learning (PyTorch 版)](https://d2l.ai) 中的以下内容：
- 第 6 章：卷积神经网络；
- 第 7 章：现代卷积神经网络。

卷积神经网络已经非常熟悉，不再赘述它的概念、优点、特性等基础知识，主要关注代码实现。CNN 的知识性内容见我的另一篇[笔记]()。


------------------------------


# CNN 的组成元素

## 卷积层



### 从头开始实现


## 汇聚层（Pooling 层）




# 流行的 CNN 架构

本节按照历史顺序介绍曾经出现并大火的 CNN 架构。本文只描述代码实现，还将重点讲述其中重要机制的实现，如残差连接、 batch normalization 等，以便自己设计网络结构时借鉴。这些网络的背景、想法等知识性内容见另一篇[笔记]()。

## LeNet


## AlexNet




## VGG



## NiN



## GoogLeNet




## ResNet



## DenseNet




