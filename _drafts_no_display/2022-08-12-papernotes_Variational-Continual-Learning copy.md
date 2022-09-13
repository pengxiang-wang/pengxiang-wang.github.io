---
title: 论文笔记：Variational Continual Learning
date: 2022-08-12
categories: [科研]
tags: [论文笔记, 持续学习]
img_path: /assets/img/
math: true
---


## 论文信息 



### [Variational Continual Learning](https://openreview.net/pdf?id=BkQqq0gRb)


- 会议：ICLR 2018
- 作者：
  - 
- 内容：从贝叶斯学派角度提出了一个持续学习框架。





--------------

# 贝叶斯观点下的持续学习

贝叶斯学派将模型参数 $$\theta$$ 当作随机变量。普通的贝叶斯监督学习只需要求一次后验分布（即一次推断），而对持续学习场景，数据是分批来的，需要根据如下迭代公式多次求后延分布：

$$p(\theta|D_{1:t}) = p(\theta |D_{1:t-1},D_t) \propto p(\theta |D_{1:t-1})p(D_t|D_{1:t-1},\theta)= p(\theta |D_{1:t-1})p(D_t|\theta)  \ t=2,\cdots,T$$
$$p(\theta|D_1) = p(\theta)p(D_1|\theta)$$

$$p(\theta)$$ 为先验分布。其中最后一个等号是假设了 $$D_t$$ 与 $$D_{t-1}$$ 独立。

# 近似算法：变分推断

变分推断是



# 防止遗忘的手段：Coreset

由于这些数据和 Online setting 不一样，不是 iid 的，必须要防止遗忘。

两种：
- Random Coreset
- K-center Coreset


更重要的意义是框架。这个 Coreset 其实不太好。

