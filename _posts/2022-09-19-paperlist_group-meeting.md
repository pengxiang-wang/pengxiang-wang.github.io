---
title: 组会论文/报告列表（长期更新）
date: 2022-09-19
categories: [科研]
tags: [日常管理, 长期更新]
img_path: /assets/img/
math: true
---

这是我组组会上讨论的论文与报告列表，按照时间倒序排序。每篇论文给出以下信息：
- 论文链接：点击论文题目即可；
- 出版信息：会议、期刊、预印本等；
- 作者：一般不详细列举，因为复制一遍这些信息实在没什么意义，只大体写一下主要作者所在的机构。仅对感兴趣的、值得关注的作详细的标注；
- 组会主讲人：均为本组博士生，以字母代替；
- 内容简介（空着的是懒得写了...）。

我的其他关于论文的博文中出现论文元信息时，也遵从上述原则。对于需要详细讲解的论文，一般不会写内容简介。




# 2022-2023 第二学期


## 2023-05-25

### [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860617.pdf)
- 会议：ECCV 2022
- 作者：东北大学，谷歌
- 主讲人：H
- 内容：


### Prompt-Tuning Few-shot learning
- 主讲人：L
- 内容：讲解几篇使用 Prompt-Tuning 的小样本学习工作的思想。


## 2023-03-02

### [Gradient Regularized Contrastive Learning for Continual Domain Adaptation]()

- 会议：AAAI 2021
- 作者：悉尼大学，香港中文大学，商汤
- 主讲人：W
- 内容：



### [Learnable istribution Calibration for Few-Shot Class-Incremental Learning](https://arxiv.org/abs/2210.00232)

- 发表：ArXiv 2022
- 作者：中国科学院大学，华为等
- 主讲人：Z
- 内容：小样本持续学习

## 2022-02-23

### [Task-Customized Self-Supervised Pre-training with Scalable Dynamic Routing](https://ojs.aaai.org/index.php/AAAI/article/view/20079)

- 会议：AAAI 2022
- 作者：华为诺亚方舟实验室
- 主讲人：L
- 内容：


### [New Insights for the Stability-Plasticity Dilemma in Online Continual Learning](https://openreview.net/forum?id=fxC7kJYwA_a)

- 会议：ICLR 2023
- 作者：首尔国立大学
- 主讲人：H
- 内容：
  - 组合了不同的 Normalization 方法，BN 负责稳定性部分，LN、IN（广泛地应用于迁移学习）负责可塑性部分。
  - 为重演样本提出了限制散开程度的损失。


## 2022-02-16

（假期进展汇报）

# 2022-2023 第一学期



## 2022-11-14

### 一个小样本任务微调的框架

- 主讲人：L
- 内容：


### [S3C: Self-Supervised Stochastic Classifiers for Few-Shot Class-Incremental Learning](https://link.springer.com/chapter/10.1007/978-3-031-19806-9_25)

- 会议：ECCV 2022
- 作者：印度科学理工学院（班加罗尔）
- 主讲人：Z
- 内容：

### 快慢网络式持续学习与任务相似性机制的结合

- 主讲人：W

## 2022-11-07

### [Cross-Domain Cross-Set Few-Shot Learning via Learning Compact and Aligned Representations](https://openreview.net/forum?id=MpJjrfSJ-Xs)

- 会议：ICLR 2022
- 主讲人：L
- 内容：

### [Temporal Latent Bottleneck: Synthesis of Fast and Slow Processing Mechanisms in Sequence Learning](https://openreview.net/forum?id=mq-8p5pUnEX)

- 会议：NIPS 2022
- 作者：蒙特利尔大学、微软、DeepMind、CIFAR 等
- 主讲人：Z
- 内容：

### [On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning](https://openreview.net/forum?id=TThSwRTt4IB)
- 发表：NIPS 2022
- 作者：意大利两所不出名大学
- 主讲人：H


## 2022-10-31

### [Worst Case Matters for Few-Shot Recognition](https://link.springer.com/chapter/10.1007/978-3-031-20044-1_6)

- 会议：ECCV 2022
- 作者：南京大学，计算机软件新技术国家重点实验室
- 主讲人：L
- 内容：

### [Do Deep Networks Transfer Invariance Across Classes?](https://openreview.net/forum?id=Fn7i_r5rR0q)

- 会议：ICLR 2022
- 作者：斯坦福大学、宾夕法尼亚大学，Finn 组
- 主讲人：Z
- 内容：

### [Compacting, Picking and Growing for Unforgetting Continual Learning](https://proceedings.neurips.cc/paper/2019/hash/3b220b436e5f3d917a1e649a0dc0281c-Abstract.html)

- 会议：NIPS 2019
- 作者：（台湾）中央研究院资讯科学研究所
- 主讲人：W
- 内容：参数隔离方法，是先训练后剪枝重新训练的 PackNet 的改进：在训练新任务时，选出旧任务参数的一部分在剪枝时也重新训练。选择哪些参数是学习了在旧任务参数上的 mask，旧任务参数是固定的，类似 Piggyback 训 mask 的方式。



## 2022-10-24

### [Curvature-Adaptive Meta-Learning for Fast Adaptation to Manifold Data](https://ieeexplore.ieee.org/abstract/document/9749838/)

- 会议/期刊：ICCV 2021, TPAMI 2022
- 作者：北京理工大学计算机学院，贾云得组
- 主讲人：L
- 内容：
  
### [Efficiently Identifying Task Groupings for Multi-Task Learning](https://openreview.net/forum?id=hqDb8d65Vfh)

- 会议：NIPS 2021
- 作者：Google、斯坦福大学，Finn 组
- 主讲人：W
- 内容：多任务学习场景的任务分组方法，基于任务相似度为任务作分组，划分模型分组训练小的多任务。任务相似度计算自训练过程的损失变化。其中任务分组、任务相似性的度量可以借鉴到持续学习上。
  
### [Exemplar-free Class Incremental Learning via Discriminative and Comparable One-class Classifiers](https://arxiv.org/abs/2201.01488)

- 发表：ArXiv 2022
- 作者：北京交通大学
- 主讲人：Z
- 内容：



## 2022-10-17

### 持续学习中区分高频/低频信息的想法

- 主讲人：H
- 内容：关于持续学习中区分高频/低频信息的想法


### [Margin-Based Few-Shot Class-Incremental Learning with Class-Level Overfitting Mitigation](https://openreview.net/forum?id=hyc27bDixNR)

- 会议：NIPS 2022
- 作者：华中科技大学，北京大学
- 主讲人：Z
- 内容：通过实验发现了持续学习在每个任务上不能学得太狠，最好学个大概即可。


### (主题)

- 论文：
  - [Free Lunch for Few-shot Learning: Distribution Calibration](https://openreview.net/forum?id=JWOiYxMG92s)
  - [Adaptive Distribution Calibration for Few-Shot Learning with Hierarchical Optimal Transport](https://openreview.net/forum?id=qOgSCLE5E8)
  - [Powering Finetuning for Few-shot Learning: Domain-Agnostic Bias Reduction with Selected Sampling](https://www.aaai.org/AAAI22Papers/AAAI-2032.TaoR.pdf)
- 会议/期刊：
  - ICLR 2021 Oral, TPAMI 2022
  - NIPS 2022
  - AAAI 2022
- 作者：
  - 悉尼科技大学
  - 香港中文大学，深圳市人工智能与机器人研究院
  - CMU
- 主讲人：L
- 内容：

### 基于 mask 的持续学习

- 论文：
  - [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf)
  - [Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.pdf)
  - [Scalable and Order-robust Continual Learning with Additive Parameter Decomposition](https://openreview.net/forum?id=r1gdj2EKPB)
  - [Supermasks in Superposition](https://proceedings.neurips.cc/paper/2020/hash/ad1f8bb9b51f023cdc80cf94bb615aa9-Abstract.html)
- 会议：
  - CVPR 2018
  - ECCV 2018
  - ICLR 2020
  - NIPS 2020
- 作者：
  - 伊利诺伊大学香槟分校
  - 伊利诺伊大学香槟分校
  - 韩国 KAIST，AITRICS
  - 华盛顿大学等
- 主讲人：W
- 内容：整理了持续学习加 Mask 的论文，为这一类方法总结出了一个分类体系（见[持续学习笔记]() 的网络结构法部分）。



## 2022-10-10

### [Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_CrossViT_Cross-Attention_Multi-Scale_Vision_Transformer_for_Image_Classification_ICCV_2021_paper.pdf)
- 会议：ICCV 2021
- 作者：MIT-IBM Watson AI Lab
- 主讲人：Z
- 内容：Cross-ViT

### [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a.html)
- 会议：ICML 2021
- 作者：Facebook
- 主讲人：Z
- 内容：DeiT

### [Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Meta-Baseline_Exploring_Simple_Meta-Learning_for_Few-Shot_Learning_ICCV_2021_paper.pdf)
- 会议：ICCV 2021
- 作者：UC San Diego，UC Berkeley 等
- 主讲人：L
- 内容：

### [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://proceedings.mlr.press/v80/serra18a.html)
- 会议：ICML 2018
- 作者：西班牙巴塞罗那的大学
- 主讲人：W
- 内容：持续学习模型 HAT，是将 mask 机制加到持续学习的第一篇论文，提出了一个很简单的、每个神经元引入一个任务 mask 的方法，并给出了训练方法，和一个解决模型容量问题的稀疏正则项，让新旧任务 mask 重合。它属于参数隔离方法，之后很多带 mask 机制的持续学习论文以此篇为基础。

### 梯度操控法持续学习

- 论文：
  - [Orthogonal Gradient Descent for Continual Learning](https://proceedings.mlr.press/v108/farajtabar20a.html)
  - [Continual Learning of Context-dependent Processing in Neural Networks](https://www.nature.com/articles/s42256-019-0080-x)
  - [Gradient Projection Memory for Continual Learning](https://openreview.net/forum?id=3AOj0RCNC2)
- 会议/期刊：
  - AISTATS 2020
  - Nature Machine Intelligence 2019
  - ICLR 2021
- 作者：
  - DeepMind
  - 中科院自动化所，类脑智能研究中心
  - 普渡大学
- 主讲人：W
- 内容：三篇基于梯度修正的持续学习论文，是这种方法最早、最经典的工作。第一、三篇把新任务的梯度投影到垂直于旧任务子空间的方向，为了防止覆盖旧任务的知识。二者的区别在第一篇直接拿旧任务用过的梯度张成子空间，第三篇是用旧任务数据（奇异值分解出的向量）构造。第二篇工作直接归结为一个修正梯度的矩阵，对其使用 RLS 算法迭代更新。

## 2022-09-19

### [Meta-attention for ViT-backed Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Xue_Meta-Attention_for_ViT-Backed_Continual_Learning_CVPR_2022_paper.pdf)
- 会议：CVPR 2022
- 作者：浙江大学、阿里
- 主讲人：Z
- 内容：

### [DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion](https://openaccess.thecvf.com/content/CVPR2022/papers/Douillard_DyTox_Transformers_for_Continual_Learning_With_DYnamic_TOken_eXpansion_CVPR_2022_paper.pdf)
- 会议：CVPR 2022
- 作者：法国索邦大学
- 主讲人：Z
- 内容：

### [A Multi-head Model for Continual Learning via Out-of-distribution Replay](https://virtual.lifelong-ml.cc/poster_33.html)
- 会议：CoLLAs 2022
- 作者：伊利诺伊大学芝加哥分校，Bing Liu 组
- 主讲人：Z
- 内容：

### [Channel Importance Matters in Few-Shot Image Classification](https://proceedings.mlr.press/v162/luo22c/luo22c.pdf)
- 会议：ICML 2022
- 作者：电子科技大学，哈尔滨工业大学
- 主讲人：L

### [Variational Continual Learning](https://openreview.net/forum?id=BkQqq0gRb)
- 会议：ICLR 2018
- 作者：剑桥大学
- 主讲人：W
- 内容：从贝叶斯学派角度提出了一个持续学习框架——变分持续学习（VCL），提出框架是主要的。同时提出了一个在此框架下简单的防止遗忘的机制——coreset。

### [Improving and Understanding Variational Continual Learning](https://arxiv.org/pdf/1905.02099.pdf)
- 发表：ArXiv 2019
- 作者：剑桥大学
- 主讲人：W
- 内容：对上一篇文章在训练技巧上做了一点改进，同时讨论了 VCL 特有的现象——剪枝效应。作者认为剪枝效应对持续学习意义是很大的


# 2021-2022 第二学期


## 2022-04-27

### [Modeling Label Space Interactions in Multi-label Classification using Box Embeddings](https://openreview.net/forum?id=tyTH9kOxcvh)
- 会议：ICLR 2022 (Poster)
- 作者：马萨诸塞大学阿默斯特分校
- 主讲人：L
- 内容：


### [Model Behavior Preserving for Class-Incremental Learning](https://ieeexplore.ieee.org/document/9705128)

- 会议：IEEE TNNLS 2022
- 作者：西安交通大学
- 主讲人：H
- 内容：


## 2022-04-20

### 数据增强论文整理

- 主讲人：L

### [Continual Learning with Recursive Gradient Optimization](https://openreview.net/forum?id=7YDLgf9_zgm)

- 会议：ICLR 2022 (Spotlight)
- 作者：清华大学计算机系
- 主讲人：W
- 内容：本文可以看成是加正则项的持续学习方法。

### [Leanring a Unified Calssifier Incrementally via Rebalancing](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf)

- 会议：CVPR 2019
- 作者：中科大、香港中文大学等
- 主讲人：Z
- 内容：


## 2022-04-13

### [MAML is a Noisy Contrastive Learner in Classification](https://openreview.net/forum?id=LDAwu17QaJz)

- 会议：ICLR 2022 (Poster)
- 作者：（台湾）国立交通大学
- 主讲人：L
- 内容：


### [The Close Relationship Between Contrastive Learning and Meta-learning](https://openreview.net/forum?id=gICys3ITSmj)

- 会议：ICLR 2022 (Poster)
- 作者：马里兰大学
- 主讲人：L
- 内容：

### [Representational Continuity for Unsupervised Continual Learning](https://openreview.net/forum?id=9Hrka5PA7LW)

- 会议：ICLR 2022 (Oral)
- 作者：
  - 纽约大学、韩国科学院、清华大学智能产业研究院等
- 主讲人：W
- 内容：这是一篇将持续学习用在无监督场景的论文，做的实验、内容还是比较综合的：里面既涉及到比较火的无监督学习模型，也把持续学习的三大类方法中比较新提出的推广到无监督场景中。目前看挺适合入门一下无监督的持续学习。无监督学习是一般是学习表示，让无监督学习持续起来，也就是题目所述的“Representational Continuity”。

## 2022-04-06

### [ConFeSS: A Framework for Single Source Cross-Domain Few-Shot Learning](https://openreview.net/forum?id=zRJu6mU2BaE)

- 会议：ICLR 2022
- 作者：高通 AI 研究院
- 主讲人：L
- 内容：

## 2022-03-16

### [(CoPE) Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams](https://openreview.net/forum?id=Tt1s9Oi1kCS)

- 会议：ICLR 2021
- 作者：比利时鲁汶大学
- 主讲人：W
- 内容：


# 2021-2022 第一学期

## 2021-12-17

### [Context-aware Attentional Pooling (CAP) for Fine-grained Visual Classification]()

- 会议：AAAI 2021
- 作者：英国边山大学
- 主讲人：L
- 内容：


## 2021-12-10

### [BNS: Building Network Structures Dynamically for Continual Learning](https://proceedings.neurips.cc/paper/2021/hash/ac64504cc249b070772848642cffe6ff-Abstract.html)
- 会议：NIPS 2021
- 作者：北大数据科学中心、胡文鹏（北大数院信息系）、王选计算机研究所、Bing Liu
- 主讲人：Z
- 内容：将强化学习用于持续学习中，在每个task中训练一个agent用来决策网络结构和初始化，使其训练后能在验证集上达到最优效果，reward包含当前task和之前task，达到防止遗忘和知识迁移两个目的。训练代价大，每个task之间agent似乎没有联系，没有真正将持续学习和强化学习联系起来。


### [Is Class-Incremental Enough for Continual Learning?](https://www.frontiersin.org/articles/10.3389/frai.2022.829842/full)
- 会议：Frontiers in AI 2022
- 作者：Andrea Cossu*, Gabriele Graffieti, Lorenzo Pellegrini, Davide Maltoni, Davide Bacciu, Antonio Carta, Vincenzo Lomonaco
- 主讲人：W
- 内容：

### [Does Continual Learning = Catastrophic Forgetting?]()
- 发表：ArXiv 2021
- 作者：Anh Thai, Stefan Stojanov, Zixuan Huang, Isaac Rehg, James M. Rehg
- 主讲人：W
- 内容：  

## 2021-12-03

### [Memory Efficient Class-Incremental Learning for Image Classification](https://ieeexplore.ieee.org/document/9422177)
- 会议：IEEE TNNLS 2021
- 作者：浙江大学计算机学院
- 主讲人：W
- 内容：

## 2021-11-26

### [IIRC: Incremental Implicitly-Refined Classification](https://openaccess.thecvf.com/content/CVPR2021/papers/Abdelsalam_IIRC_Incremental_Implicitly-Refined_Classification_CVPR_2021_paper.pdf)
- 会议：CVPR 2021
- 作者：蒙特利尔大学等
- 主讲人：Z
- 内容：提出了持续学习中出现不同粒度的类别，且相互关系未知。用多标签分类的指标作为评价标准，在几个经典算法上观察了实验效果，说明了粗细粒度的关系会影响分类效果。
  
### [HCV: Hierarchy-Consistency Verification for Incremental Implicitly-Refined Classification](https://www.bmvc2021-virtualconference.com/assets/papers/0008.pdf)
- 会议：BMVC 2021
- 作者：西班牙巴塞罗那的大学，南开大学
- 主讲人：Z
- 内容：针对IIRC问题提出了判断类别关系的方法，根据前面类别的打分确定是否为之前某一类的子类。超类和子类同时输出高分。测试时根据训练得到的层级关系调整矛盾的预测结果。


## 2021-11-19

### [Overcoming Catastrophic Forgetting in Incremental Few-Shot Learning by Finding Flat Minima](https://openreview.net/forum?id=ALvt7nXa2q)
- 会议：NIPS 2021
- 作者：香港科技大学
- 主讲人：H
- 内容：


### [Intriguing Properties of Contrastive Losses](https://openreview.net/forum?id=rYhBGWYm6AU)
- 会议：NIPS 2021
- 作者：Google
- 主讲人：L
- 内容：

## 2021-11-12


### [Efficiently Identifying Task Groupings for Multi-Task Learning](https://openreview.net/forum?id=hqDb8d65Vfh)
- 会议：NIPS 2021
- 作者：Google、斯坦福大学，Finn 组
- 主讲人：Z
- 内容：
  - 目的：设计高效的多任务分组方法。
  - 方法：提出 inter-task affinity，用a任务的梯度方向观察b任务的损失函数变化情况，以此刻画任务间的相关程度。
  - 主要结论：此算法和SOTA相比在测试准确率不降低的情况下大幅减少了计算时间。


### [Meta-learning with an Adaptive Task Scheduler](https://openreview.net/forum?id=MTs2adH_Qq)
- 会议：NIPS 2021
- 作者：斯坦福大学，中科大，腾讯 AI Lab 等，Finn 组
- 主讲人：H
- 内容：


## 2022-11-04

### [Can multi-label classification networks know what they don’t know?](https://openreview.net/forum?id=enKhMfthDFS)
- 会议：NIPS 2021
- 作者：CMU 等
- 主讲人：L
- 内容：启发于基于能量的OOD判别方法，本文针对多标签分类问题基于能量模型提出一种OOD鉴别指标。


## 2022-10-29


### [Few-shot Open-set Recognition by Transformation Consistency](https://openaccess.thecvf.com/content/CVPR2021/papers/Jeong_Few-Shot_Open-Set_Recognition_by_Transformation_Consistency_CVPR_2021_paper.pdf)
- 会议：CVPR 2021
- 作者：韩国 KAIST
- 主讲人：L
- 内容：提出一种不需要训练集中包含未知样本的小样本开放集识别的方法。这种方法基于一类通过对类别原型进行变换的小样本识别方法，利用这种变换的一致性，通过取代原型的方法比较取代前后的距离来判断是否是unseen样本。

### [A continual learning survey: Defying forgetting in classification tasks](https://arxiv.org/abs/1909.08383)
- 会议：IEEE TPAMI 2021
- 作者：比利时鲁汶大学，西班牙巴塞罗那的大学，华为诺亚方舟实验室等
- 主讲人：W
- 内容：


## 2021-10-22

### [Co2L：Contrastive Continual Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Cha_Co2L_Contrastive_Continual_Learning_ICCV_2021_paper.pdf)
- 会议：ICCV 2021
- 作者：韩国 KAIST
- 主讲人：Z
- 内容：
  - 目的：将对比学习用于持续学习中。
  - 方法：1.将每个task的交叉熵损失换成监督型对比损失，并提出非对称损失，防止进一步分离之前见过的类（否则会出现存储样本与整体分布的偏差）。2.用instancewise relation distillation防止遗忘。
  - 主要结论：对比损失能提取更适合在任务间迁移的特征。

### [Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks](https://openreview.net/pdf?id=OdklztJBBYH)
- 会议：NIPS 2021
- 作者：墨尔本大学，北大人工智能学院等
- 主讲人：H
- 内容：在 WRN32-10 的下框架，探究怎样的结构有助于提高神经网络的对抗鲁棒性，发现当越靠近输出层宽度越小时，网络的对抗鲁棒性越强，当越远离输出层的网络参数越大时，对抗鲁棒性越强。

## 2022-10-08

### [Few-Shot Learning with Part Discovery and Augmentation from Unlabeled Images](https://www.ijcai.org/proceedings/2021/0313.pdf)
- 会议：IJCAI 2021
- 作者：中科大，中科院计算所
- 主讲人：L
- 内容：本文解决的小样本问题场景为：大量的无标签数据可作为特征提取器的预训练数据集。核心思想就是图像中关键Part的获取。在预训练部分，选择每张图像中信息量最大的Part，利用这一Part参与对比学习从而得到下游任务需要的特征提取器。在下游任务中，先用小样本数据训练一个分类器，再用该分类器对无标签数据进行预分类，选择分类概率较高的样本作为增强数据。增强的方法即通过一个样本的attention block，意在放大与类别相关的特征。最后再用增强的数据及原小样本数据重新训练分类器。

### [Data Augmentation for Meta-Learning](https://proceedings.mlr.press/v139/ni21a.html)
- 会议：ICML 2021
- 作者：马里兰大学
- 主讲人：H
- 内容：本文探索了在元学习的不同阶段做数据增广，观察在每个阶段做数据增广的不同表现，得到数据增广在元学习不同阶段所能起到的不同作用



## 2022-09-17

### [Continual Learning in the Teacher-Student Setup: Impact of Task Similarity](http://proceedings.mlr.press/v139/lee21e/lee21e.pdf)
- 会议：ICML 2021
- 作者：帝国理工大学，牛津大学等
- 主讲人：Z
- 内容：
  - 目的：实验观察持续学习中前后任务的相似性对结果的影响。仅涉及两层神经网络两个任务的情况。
  - 方法：teacher-student setup，ODE数值模拟。
  - 主要结论：intermediate task similarity leads to greatest forgetting


### 持续且无遗忘的深度学习方法研究
- 会议：博士学位论文
- 作者：胡文鹏（北大数院信息系）
- 主讲人：W
- 内容：
  - 目的：通过持续学习克服灾难性遗忘问题，从表面原因（训练样本分布不均衡）与根本原因（特征偏置）下手
  - 本次主要汇报第3部分：
    - 1.参数生成与模型自适应方法（PGMA）：一种持续学习算法，针对表面原因
    - 2.全面学习（HL）：解决单类别分类，用全面正则项（H-reg）实现，解决根本原因
    - 3.全面持续学习框架：一种持续学习算法，结合H-reg，引入参数迁移、后处理机制，解决根本原因


