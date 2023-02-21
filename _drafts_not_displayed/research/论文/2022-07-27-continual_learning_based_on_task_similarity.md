---
title: 基于任务相似性的持续学习
date: 2022-07-27
categories: [科研]
tags: [学习笔记, 持续学习]
img_path: /assets/img/
math: true
---

有些持续学习算法会考虑**任务相似性**，对相似与不相似的任务采取不同的训练方法。任务相似性机制与防遗忘机制是独立的，因此很多现有的持续学习算法都可以引入任务相似性机制。

任务相似性机制关注两方面：

- 如何度量任务相似性，以及如何计算；
- 任务相似性如何作用于训练算法。

任务相似性机制也在多任务学习、元学习等其他相关领域被提出，因此本文也会涉及一些其他领域中的任务相似性机制，以借鉴到持续学习场景中。



多任务学习与持续学习的区别在于是不是 online setting。所以对于相似度也要随任务迭代地。

在多任务学习，task grouping 本质上为了缩减指数级别的搜索算法。


判断相似度的方法有很多，可以 if-else 硬性地判断，（相似度 0 或 1），也可以软式。。既可以用固定的算两个分布（数据）的距离，也可以利用训练过程中的得出的指标。

对于二元的相似度，涉及到相似关系。试问是否为等价关系？如何处理？有没有传递性？在 CAT 这篇文章中，没有传递性。




## 分布距离

KL 散度

Wasserstein 距离
f-divergence: Hellinger distance, total variation distance
MMD（Maximum Mean Discrepancy）

https://en.wikipedia.org/wiki/Statistical_distance


度量学习（metric learning）来学习距离

https://www.zhihu.com/question/39872326

A principled approach for learning task similarity in multitask learning

\begin{itemize}
    \item 会议：IJCAI 2019
    \item 作者：Changjian Shui,Mahdieh Abbasi,Louis-Emile Robitaille, Boyu Wang, Christian Gagne
    \item 机构：Université Laval，University of Pennsylvania
\end{itemize}

## 在训练中加入





# 论文例子


##  CAT





# 如何与 mask 机制结合？

参考 Task Grouping 论文 PPT 最后一页怎么用到持续学习