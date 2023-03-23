---
title: 持续学习论文列表（长期更新）
date: 2022-04-10
categories: [科研]
tags: [持续学习, 长期更新]
img_path: /assets/img/
math: true
---

本文汇总了目前我看过的持续学习领域相关文献。按照我的[《持续学习基础知识》]()一文的脉络整理。


----------
**目录**

* TOC
{:toc}


---------

对一些重要的论文，标识。

影响力：
个人推荐指数：


# 综述

### [Lifelong Learning Algorithms](https://link.springer.com/book/10.1007/978-1-4615-5529-2)
- 图书：《Learning to learn》，1998，第 8 章
- 作者：[Sebastian Thrun](https://zh.wikipedia.org/zh-cn/塞巴斯蒂安·特龙)
- 内容：`《Learning to learn》这本书是元学习领域的开山鼻祖，Thrun 是元学习和持续学习早期最重要的研究者之一。这本书主要是系统地引入了元学习的概念，中间穿插总结了当时多任务学习和持续学习的算法。（我一直没办法看到这本书，那个 Springer 用学校登录不上，要花钱 TAT）`
- 

### [A continual learning survey: Defying forgetting in classification tasks](https://ieeexplore.ieee.org/document/9349197)
- 期刊：TPAMI 2021
- 作者：比利时鲁汶大学：Matthias De Lange, Rahaf Aljundi；西班牙 UAB：Marc Masana；Sarah Parisot, Xu Jia, Ales Leonardis, Gregory Slabaugh, Tinne Tuytelaars
- 内容：


### [Lifelong Machine Learning](https://www.cs.uic.edu/~liub/lifelong-machine-learning.html)
- 图书：出版社，第 1 版 时间，第 2 版时间
- 作者：
  - Zhiyuan Chen
  - Bing Liu
- 内容：


### [Reviewing continual learning from the perspective of human-level intelligence](https://arxiv.org/abs/2111.11964)
- 期刊：
- 作者：Yifan Chang, Wenbo Li, Jian Peng, Bo Tang, Yu Kang, Yinjie Lei, Yuanmiao Gui, Qing Zhu, Yu Liu, Haifeng Li
- 内容：


# 大概念/理论

### [CHILD: A First Step Towards Continual Learning](https://link.springer.com/content/pdf/10.1023/A:1007331723572.pdf)
- 期刊：Machine Learning 1997
- 作者：Mark B. Ring (德国国家信息技术研究中心, GMD)
- 内容：The first paper to introduce CL


### [Is Learning The n-th Thing Any Easier Than Learning The First?](https://proceedings.neurips.cc/paper/1995/file/bdb106a0560c4e46ccc488ef010af787-Paper.pdf)
- 会议：NIPS 1995
- 作者：Sebastian Thrun
- 内容：

# 数据集与指标





# 防遗忘机制：重演方法

## 真重演数据

### [iCaRL: Incremental Classifier and Representation Learning](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)
- 会议：CVPR 2017
- 作者：Sylvestre-Alvise Rebuffi, et al, University of Oxford/IST Austria
- 内容：idea
  - Aims to learn a strong data representation
  - Nearest-Mean-of-Exemplars Classification, examplars take part in prediction
  - Simply fix memory size, allocate to each task averagely


### [Gradient Episodic Memory for Continual Learning](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)
- 会议：NIPS 2017
- 作者：David Lopez-Paz, et al, Facebook
- 内容：GEM


### [Efficient Lifelong Learning with A-GEM](https://openreview.net/forum?id=Hkf2_sC5FX)
- 会议：ICLR 2018
- 作者：Arslan Chaudhry, et al, University of Oxford, Facebook
- 内容：Idea
  - Exploits exemplars to solve a constrained optimization problem
  - Store previous task gradient
  - A-GEM propose a small change to the loss function which makes GEM orders of magnitude faster at training time while maintaining similar performance


### [Experience Replay for Continual Learning](https://papers.nips.cc/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)
- 会议：NIPS 2019
- 作者：D. Rolnick, et al, UPenn, DeepMind
- 内容：CLEAR
  - Actor-critic training on a mixture of new and replayed experiences
  - Off-policy learning and behavioral cloning from replay to enhance stability
  - On-policy learning to preserve plasticity
  - Off-policy: V-Trace learning algorithm


### [Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams](https://openaccess.thecvf.com/content/ICCV2021/papers/De_Lange_Continual_Prototype_Evolution_Learning_Online_From_Non-Stationary_Data_Streams_ICCV_2021_paper.pdf)
- 会议：ICCV 2021
- 作者：M De Lange, et al, KU Leuven
- 内容：


### [Continual Learning with Recursive Gradient Optimization](https://openreview.net/forum?id=7YDLgf9_zgm)
- 会议：ICLR 2022 (Spotlight)
- 作者：
    - Hao Liu - 清华大学计算机系（可能是后者的学生）
    - [刘华平](https://www.cs.tsinghua.edu.cn/info/1122/3566.htm) - 清华大学计算机系
- 内容：本文可以看成是加正则项的持续学习方法。



## 伪重演数据（生成式）

### [Continual Learning with Deep Generative Replay](https://papers.nips.cc/paper/2017/file/0efbe98067c6c73dba1250d2beaa81f9-Paper.pdf)
- 会议：NIPS 2017
- 作者：Hanul Shin, et al, MIT, SK T-Brain
- 内容：DGR
  - Introduces GAN
  - Dual model: Generator \& Solver
  - Data for previous tasks can easily be sampled



# 防遗忘机制：正则化方法

### [Learning without Forgetting](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37?ref=https://codemonkey.link)
- 期刊：ECCV 2016 / IEEE TPAMI 2017
- 作者：Zhizhong Li, et al, University of Illinois Urbana Champaign
- 内容：LwF

### [Overcoming Catastrophic Forgetting in Neural Networks](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- 期刊：PNAS 2017
- 作者：James Kirkpatrick, et al, DeepMind, Imperial College London
- 内容：EWC
When training task B after task A, minimize $\mathcal{L}$ instead:
$$
\mathcal{L}(\theta)=\mathcal{L}_{B}(\theta)+\sum_{i} \frac{\lambda}{2} F_{i,i}\left(\theta_{i}-\theta_{A, i}^{*}\right)^{2}
$$
Constrain important parameters to stay close to their old values

### [Overcoming Catastrophic Forgetting by Incremental Moment Matching](https://proceedings.neurips.cc/paper/2017/file/f708f064faaf32a43e4d3c784e6af9ea-Paper.pdf)
- 会议：NIPS 2017
- 作者：Sang-Woo Lee, et al, Seoul National University
- 内容：IMM   incrementally matches the moment of the posterior distribution of the network
 averages the parameters of two networks in each layer, using mixing ratios $\alpha_k$ with
$\sum_k^K \alpha_k = 1$ 

### [Better Weight Consolidation and Less Catastrophic Forgetting](https://arxiv.org/pdf/1802.02950.pdf)
- 期刊：IEEE ICPR 2017
- 作者：Xialei Liu, et al, UAB Spain, University of Florence
- 内容：R-EWC: Rotate your networks: A factorized rotation of parameter space in conjunction with EWC

### [SI: Continual learning through Synaptic Intelligence](https://arxiv.org/pdf/1703.04200.pdf)
**没有导入**
- 会议：ICML 2017
- 作者：Friedemann Zenke, et al, Stanford University
- 内容：Bring biological complexity into artificial neural networks

### [Memory Aware Synapses: Learning What (not) to Forget](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf)
- 会议：ECCV 2018
- 作者：Rahaf Aljundi, et al, KU Leuven, Facebook
- 内容：MAS       Accumulates an importance measure for each parameter of the net- work, based on how sensitive the predicted output function is to a change in this parameter. When learning a new task, changes to important parameters can then be penalized, effectively preventing important knowledge related to previous tasks from being overwritten. 



### [Variational Continual Learning](https://openreview.net/pdf?id=BkQqq0gRb), [Improving and Understanding Variational Continual Learning](https://arxiv.org/pdf/1905.02099.pdf)
- 会议：ICLR 2018, arXiv 2019
- 作者：Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, Richard E. Turner; Siddharth Swaroop∗, Cuong V. Nguyen∗, Thang D. Bui†, Richard E. Turner∗
- 内容：从贝叶斯学派角度提出了一个持续学习框架——变分持续学习（VCL），提出框架是主要的。同时提出了一个在此框架下简单的防止遗忘的机制——coreset。第二篇是对上一篇文章在训练技巧上做了一点改进，同时讨论了 VCL 特有的现象——剪枝效应。作者认为剪枝效应对持续学习意义是很大的。




# 防遗忘机制：网络结构方法

### [Expert Gate: Lifelong Learning with a Network of Experts](https://openaccess.thecvf.com/content_cvpr_2017/papers/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.pdf)
- 会议：CVPR 2017
- 作者：Rahaf Aljundi, et al, KU Leuven
- 内容：

### [PathNet: Evolution channels gradient descent in super neural networks](https://arxiv.org/pdf/1701.08734.pdf)
- 会议：arxiv 2017
- 作者：Chrisantha Fernando, et al, DeepMind
- 内容：
  - Agents embedded in network to discover which parameter to re-use for new tasks
  - Pathways through network are the subset of parameters 
  - These parameters updated by the forwards and backwards passes of the backpropogation algorith

### [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf)
- 会议：CVPR 2018
- 作者：Arun Mallya, et al, University of Illinois at Urbana-Champaign
- 内容：
    - Network Pruning: free up redundant parameters that can then be employed to learn new tasks
    - Sequentially “pack” multiple tasks into a single network



### [Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks](https://proceedings.neurips.cc/paper/2020/file/d7488039246a405baf6a7cbc3613a56f-Paper.pdf)
- 会议：NIPS 2020
- 作者：
  - [Zixuan Ke](https://underline.io/speakers/97701-zixuan-ke)：伊利诺伊大学芝加哥分校，博士生，后者的学生。
  - [Bing Liu](https://www.cs.uic.edu/~liub/)：伊利诺伊大学芝加哥分校，教授。他是《终身机器学习》的作者，我有系列[读书笔记](https://pengxiang-wang.github.io/tags/终身机器学习/)。
  - [Xingchang Huang](https://people.mpi-inf.mpg.de/~xhuang/)：苏黎世联邦理工大学，博士生。
- 内容：提出了一个持续学习模型。它不仅关注不相似任务的灾难性遗忘，还关注相似任务的知识迁移。本文是持续学习与迁移学习相结合。



# 拓展方向

## 多粒度持续学习


### [IIRC: Incremental Implicitly-Refined Classification](https://openaccess.thecvf.com/content/CVPR2021/papers/Abdelsalam_IIRC_Incremental_Implicitly-Refined_Classification_CVPR_2021_paper.pdf)


- 会议：CVPR 2021
- 作者：
    - Mohamed Abdelsalam, Mojtaba Faramarzi - 蒙特利尔大学，[Mila-Quebec AI Institute](https://mila.quebec)，博士生
    - [Shagun Sodhani](https://shagunsodhani.com) - Facebook
    - [Sarath Chandar](http://sarathchandar.in) - 蒙特利尔工程学院，加拿大高等研究院（CIFAR），[Mila-Quebec AI Institute](https://mila.quebec)
- 内容：提了一个持续学习的新场景，是类别增量学习的推广：允许数据包含多个由粗到细的标签，允许旧数据携带细标签进入新任务。


## 无监督持续学习



### [Representational Continuity for Unsupervised Continual Learning](https://openreview.net/forum?id=9Hrka5PA7LW)


- 会议：ICLR 2022 (Oral)
- 作者：
    - [Divyam Madaan*](https://dmadaan.com) - 纽约大学，博士生
    - Jaehong Yoon - 韩科院，博士生（微软实习发的文章）
    - [李元春](https://yuanchun-li.github.io/) - [清华大学智能产业研究院](https://air.tsinghua.edu.cn)
    - [刘云新](https://yunxinliu.github.io/) - [清华大学智能产业研究院](https://air.tsinghua.edu.cn)
    - [Sung Ju Hwang](http://www.sungjuhwang.com) - 韩科院，前两人的导师
- 内容：这是一篇将持续学习用在无监督场景的论文，做的实验、内容还是比较综合的：里面既涉及到比较火的无监督学习模型，也把持续学习的三大类方法中比较新提出的推广到无监督场景中。目前看挺适合入门一下无监督的持续学习。无监督学习是一般是学习表示，让无监督学习持续起来，也就是题目所述的“Representational Continuity”。


## 

