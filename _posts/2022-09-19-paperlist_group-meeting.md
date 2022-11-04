---
title: 组会论文/报告列表（长期更新）
date: 2022-09-19
categories: [科研]
tags: [长期更新]
img_path: /assets/img/
math: true
---

> 说明：这是我所在研究组组会上讲的论文列表，按照时间倒序排序。主讲人均为本组博士生。

# 2022-2023 上半年


## 2022-10-31

### [Worst Case Matters for Few-Shot Recognition](https://link.springer.com/chapter/10.1007/978-3-031-20044-1_6)

- 会议：ECCV 2022
- 作者：Minghao Fu, Yun-Hao Cao, Jianxin Wu
- 主讲人：L
- 内容：

### [Do Deep Networks Transfer Invariance Across Classes?](https://openreview.net/pdf?id=Fn7i_r5rR0q)

- 会议：ICLR 2022
- 作者：Allan Zhou∗ & Fahim Tajwar, Alexander Robey, Tom Knowles, George J. Pappas & Hamed Hassani, Chelsea Finn
- 主讲人：Z
- 内容：

### [Compacting, Picking and Growing for Unforgetting Continual Learning](https://proceedings.neurips.cc/paper/2019/hash/3b220b436e5f3d917a1e649a0dc0281c-Abstract.html)

- 会议：NIPS 2019
- 作者：Ching-Yi Hung, Cheng-Hao Tu, Cheng-En Wu, Chien-Hung Chen, Yi-Ming Chan, Chu-Song Chen
- 主讲人：W
- 内容：参数隔离方法，是先训练后剪枝重新训练的 PackNet 的改进：在训练新任务时，选出旧任务参数的一部分在剪枝时也重新训练。选择哪些参数是学习了在旧任务参数上的 mask，旧任务参数是固定的，类似 Piggyback 训 mask 的方式。



## 2022-10-24

### [Curvature-Adaptive Meta-Learning for Fast Adaptation to Manifold Data](https://ieeexplore.ieee.org/abstract/document/9749838/)

- 会议/期刊：ICCV 2021, TPAMI 2022
- 作者：Zhi Gao, Yuwei Wu, Mehrtash T Harandi, Yunde Jia
- 主讲人：L
- 内容：
  
### [Efficiently Identifying Task Groupings for Multi-Task Learning](https://openreview.net/forum?id=hqDb8d65Vfh)

- 会议：NIPS 2021
- 作者：Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, Chelsea Finn
- 主讲人：W
- 内容：多任务学习场景的任务分组方法，基于任务相似度为任务作分组，划分模型分组训练小的多任务。任务相似度计算自训练过程的损失变化。其中任务分组、任务相似性的度量可以借鉴到持续学习上。
  
### [Exemplar-free Class Incremental Learning via Discriminative and Comparable One-class Classifiers](https://arxiv.org/abs/2201.01488)

- 发表：arXiv 2022
- 作者：Wenju Sun, Qingyong Li, Jing Zhang, Danyu Wang, Wen Wang, Yangli-ao Geng
- 主讲人：Z
- 内容：





## 2022-10-17

### (主题)

- 主讲人：H
- 内容：关于持续学习中区分高频/低频信息的想法


### [Margin-Based Few-Shot Class-Incremental Learning with Class-Level Overfitting Mitigation](https://arxiv.org/abs/2210.04524)

- 会议：NIPS 2022
- 作者：Yixiong Zou, Shanghang Zhang, Yuhua Li, Ruixuan Li
- 主讲人：Z
- 内容：通过实验发现了持续学习在每个任务上不能学得太狠，最好学个大概即可。


### (主题)

- 论文：
  - [Free Lunch for Few-shot Learning: Distribution Calibration](https://openreview.net/forum?id=JWOiYxMG92s)
  - [Adaptive Distribution Calibration for Few-Shot Learning with Hierarchical Optimal Transport](https://arxiv.org/abs/2210.04144)
  - [Powering Finetuning for Few-shot Learning: Domain-Agnostic Bias Reduction with Selected Sampling](https://www.aaai.org/AAAI22Papers/AAAI-2032.TaoR.pdf)
- 会议：
  - ICLR 2021 Oral, TPAMI 2022
  - NIPS 2022
  - AAAI 2022
- 作者：
  - Shuo Yang, Lu Liu, Min Xu
  - Dandan Guo, Long Tian, He Zhao, Mingyuan Zhou, Hongyuan Zha
  - Ran Tao, Han Zhang, Yutong Zheng, Marios Savvides
- 主讲人：L
- 内容：

### 基于 mask 的持续学习

- 论文：
  -  [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf)
  - [Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.pdf)
  - [Scalable and Order-robust Continual Learning with Additive Parameter Decomposition](https://openreview.net/pdf?id=r1gdj2EKPB), 
  - [Supermasks in Superposition](https://proceedings.neurips.cc/paper/2020/file/ad1f8bb9b51f023cdc80cf94bb615aa9-Paper.pdf),
- 会议：
  - CVPR 2018
  - ECCV 2018
  - ICLR 2020
  - NIPS 2020
- 作者：
  - Arun Mallya, Svetlana Lazebnik
  - Arun Mallya, Dillon Davis, Svetlana Lazebnik
  - Jaehong Yoon, Saehoon Kim, Eunho Yang, Sung Ju Hwang
  - Mitchell Wortsman, Vivek Ramanujan,  Rosanne Liu,, Aniruddha Kembhavi, Mohammad Rastegari, Jason Yosinski, Ali Farhadi
- 主讲人：W
- 内容：整理了持续学习加 Mask 的论文，为这一类方法总结出了一个分类体系（见[持续学习笔记]() 的网络结构法部分）。



## 2022-10-10

### [Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_CrossViT_Cross-Attention_Multi-Scale_Vision_Transformer_for_Image_Classification_ICCV_2021_paper.pdf)
- 会议：ICCV 2021
- 作者：Chun-Fu (Richard) Chen, Quanfu Fan, Rameswar Panda (MIT-IBM Watson AI Lab)
- 主讲人：Z
- 内容：Cross-ViT

### [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a.html)
- 会议：ICML 2021
- 作者：Chun-Fu (Richard) Chen, Quanfu Fan, Rameswar Panda (MIT-IBM Watson AI Lab)
- 主讲人：Z
- 内容：DeiT

### [Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Meta-Baseline_Exploring_Simple_Meta-Learning_for_Few-Shot_Learning_ICCV_2021_paper.pdf)
- 会议：ICCV 2021
- 作者：Yinbo Chen (UC San Diego), Zhuang Liu (UC Berkeley), Huijuan Xu (Penn State University), Trevor Darrell (UC Berkeley), Xiaolong Wang (UC San Diego)
- 主讲人：L
- 内容：

### [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://proceedings.mlr.press/v80/serra18a.html)
- 会议：ICML 2018
- 作者：Joan Serrà, Dídac Surís, Marius Miron, Alexandros Karatzoglou
- 主讲人：W
- 内容：持续学习模型 HAT，是将 mask 机制加到持续学习的第一篇论文，提出了一个很简单的、每个神经元引入一个任务 mask 的方法，并给出了训练方法，和一个解决模型容量问题的稀疏正则项，让新旧任务 mask 重合。它属于参数隔离方法，之后很多带 mask 机制的持续学习论文以此篇为基础。

### 1.[Orthogonal Gradient Descent for Continual Learning](https://arxiv.org/abs/1910.07104), 2.[Continual Learning of Context-dependent Processing in Neural Networks](https://www.nature.com/articles/s42256-019-0080-x), 3.[Gradient Projection Memory for Continual Learning](https://openreview.net/forum?id=3AOj0RCNC2)
- 期刊/会议：1. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics; 2. Nature Machine Intelligence 2019; 3. ICLR 2021
- 作者：1. Mehrdad Farajtabar, Navid Azizan, Alex Mott, Ang Li; 2. Guanxiong Zeng, Yang Chen, Bo Cui, Shan Yu; 3. Gobinda Saha, Isha Garg, Kaushik Roy
- 主讲人：W
- 内容：三篇基于梯度修正的持续学习论文，是这种方法最早、最经典的工作。第一、三篇把新任务的梯度投影到垂直于旧任务子空间的方向，为了防止覆盖旧任务的知识。二者的区别在第一篇直接拿旧任务用过的梯度张成子空间，第三篇是用旧任务数据（奇异值分解出的向量）构造。第二篇工作直接归结为一个修正梯度的矩阵，对其使用 RLS 算法迭代更新。

## 2022-09-19

[Meta-attention for ViT-backed Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Xue_Meta-Attention_for_ViT-Backed_Continual_Learning_CVPR_2022_paper.pdf)
- 会议：CVPR 2022
- 作者：Mengqi Xue, Haofei Zhang, Jie Song, Mingli Song
- 主讲人：Z

[DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion](https://openaccess.thecvf.com/content/CVPR2022/papers/Douillard_DyTox_Transformers_for_Continual_Learning_With_DYnamic_TOken_eXpansion_CVPR_2022_paper.pdf)
- 会议：CVPR 2022
- 作者：Arthur Douillard, Alexandre Ramé, Guillaume Couairon, Matthieu Cord
- 主讲人：Z

[A Multi-head Model for Continual Learning via Out-of-distribution Replay](https://arxiv.org/pdf/2208.09734.pdf)
- 会议：CoLLAs 2022
- 作者：Gyuhak Kim, Zixuan Ke, Bing Liu
- 主讲人：Z

[Channel Importance Matters in Few-Shot Image Classification](https://proceedings.mlr.press/v162/luo22c/luo22c.pdf)
- 会议：ICML 2022
- 作者：Xu Luo, Jing Xu, Zenglin Xu
- 主讲人：L

### [Variational Continual Learning](https://openreview.net/pdf?id=BkQqq0gRb)
- 会议：ICLR 2018
- 作者：Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, Richard E. Turner
- 主讲人：W
- 内容：从贝叶斯学派角度提出了一个持续学习框架——变分持续学习（VCL），提出框架是主要的。同时提出了一个在此框架下简单的防止遗忘的机制——coreset。

### [Improving and Understanding Variational Continual Learning](https://arxiv.org/pdf/1905.02099.pdf)
- 会议：2019 (ArXiv)
- 作者：Siddharth Swaroop, Cuong V. Nguyen, Thang D. Bui†, Richard E. Turner
- 主讲人：W
- 内容：对上一篇文章在训练技巧上做了一点改进，同时讨论了 VCL 特有的现象——剪枝效应。作者认为剪枝效应对持续学习意义是很大的


# 2021-2022 下学年



## 2022-04-27

### [Modeling Label Space Interactions in Multi-label Classification using Box Embeddings](https://openreview.net/forum?id=tyTH9kOxcvh)
- 会议：ICLR 2022 (Poster)
- 作者：Dhruvesh Patel, Pavitra Dangati, Jay-Yoon Lee, Michael Boratko, Andrew McCallum
- 主讲人：L




### [Model Behavior Preserving for Class-Incremental Learning](https://ieeexplore.ieee.org/document/9705128)


- 会议：IEEE TNNLS 2022
- 作者：Yu Liu, Xiaopeng Hong , Xiaoyu Tao , Songlin Dong , Jingang Shi , Yihong Gong
- 主讲人：H


## 2022-04-20

### 数据增强论文整理
- 主讲人：L


### [Continual Learning with Recursive Gradient Optimization](https://openreview.net/forum?id=7YDLgf9_zgm)


- 会议：ICLR 2022 (Spotlight)
- 作者：
    - Hao Liu - 清华大学计算机系（可能是后者的学生）
    - [刘华平](https://www.cs.tsinghua.edu.cn/info/1122/3566.htm) - 清华大学计算机系
- 主讲人：W
- 内容：本文可以看成是加正则项的持续学习方法。

### [Leanring a Unified Calssifier Incrementally via Rebalancing]

- 会议：CVPR 2019
- 作者：
    - [侯赛辉](https://hshustc.github.io) - 中科大，博士生
    - [Xinyu Pan](https://hk.linkedin.com/in/xinyu-pan) - 香港中文大学，博士生
    - [吕健勤](https://www.mmlab-ntu.com/person/ccloy/) - 南洋理工大学[MMLab](https://www.mmlab-ntu.com)，副教授
    - [王子磊](http://staff.ustc.edu.cn/~zlwang) - 中科大[VIM研究组](http://vim.ustc.edu.cn)，助理教授
    - [林达华](http://dahua.site) - 香港中文大学，助理教授
- 主讲人：Z



## 2022-04-13


### [MAML is a Noisy Contrastive Learner in Classification](https://openreview.net/forum?id=LDAwu17QaJz)
- 会议：ICLR 2022 (Poster)
- 作者：
    - [高家祥](https://iandrover.github.io) - 博士生
    - [邱維辰](https://walonchiu.github.io) - (台湾)国立交通大学，前一人的导师
    - [陳品諭](https://www.pinyuchen.com) - IBM研究院，前一人的导师
- 主讲人：L



### [The Close Relationship Between Contrastive Learning and Meta-learning](https://openreview.net/forum?id=gICys3ITSmj)
- 会议：ICLR 2022 (Poster)
- 作者：
    - 倪仁坤，Manli Shu - 马里兰大学，博士生
    - [Hossein Souri](https://hsouri.github.io) - 约翰霍普金斯大学，博士生
    - [Micah Goldblum](https://goldblum.github.io) - 马里兰大学，博士后
    - [Tom Goldstein](https://www.cs.umd.edu/~tomg/) - 马里兰大学，导师
- 主讲人：L


### [Representational Continuity for Unsupervised Continual Learning](https://openreview.net/forum?id=9Hrka5PA7LW)
- 会议：ICLR 2022 (Oral)
- 作者：
    - [Divyam Madaan*](https://dmadaan.com) - 纽约大学，博士生
    - Jaehong Yoon - 韩科院，博士生（微软实习发的文章）
    - [李元春](https://yuanchun-li.github.io/) - [清华大学智能产业研究院](https://air.tsinghua.edu.cn)
    - [刘云新](https://yunxinliu.github.io/) - [清华大学智能产业研究院](https://air.tsinghua.edu.cn)
    - [Sung Ju Hwang](http://www.sungjuhwang.com) - 韩科院，前两人的导师
- 主讲人：W
- 内容：这是一篇将持续学习用在无监督场景的论文，做的实验、内容还是比较综合的：里面既涉及到比较火的无监督学习模型，也把持续学习的三大类方法中比较新提出的推广到无监督场景中。目前看挺适合入门一下无监督的持续学习。无监督学习是一般是学习表示，让无监督学习持续起来，也就是题目所述的“Representational Continuity”。

## 2022-04-06

### [ConFeSS: A Framework for Single Source Cross-Domain Few-Shot Learning](https://openreview.net/forum?id=zRJu6mU2BaE)
- 会议：ICLR 2022
- 作者：Debasmit Das, Sungrack Yun, Fatih Porikli
- 主讲人：L

## 2022-03-16

### [(CoPE) Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams](https://openreview.net/forum?id=Tt1s9Oi1kCS)
- 会议：ICLR 2021
- 作者：Matthias De Lange,Tinne Tuytelaars
- 主讲人：W



# 2021-2022 上学年

## 2021-12-17

### [Context-aware Attentional Pooling (CAP) for Fine-grained Visual Classification](https://arxiv.org/abs/2101.06635)
- 会议：AAAI 2021
- 作者：Ardhendu Behera, Zachary Wharton, Pradeep Hewage, Asish Bera
- 主讲人：L


## 2021-12-10



### [BNS: Building Network Structures Dynamically for Continual Learning](https://proceedings.neurips.cc/paper/2021/file/ac64504cc249b070772848642cffe6ff-Paper.pdf)
- 会议：NIPS 2021
- 作者：Qi Qin, Han Peng, Wenpeng Hu, Dongyan Zhao, Bing Liu
- 主讲人：Z
- 内容：将强化学习用于持续学习中，在每个task中训练一个agent用来决策网络结构和初始化，使其训练后能在验证集上达到最优效果，reward包含当前task和之前task，达到防止遗忘和知识迁移两个目的。训练代价大，每个task之间agent似乎没有联系，没有真正将持续学习和强化学习联系起来。


### [Is Class-Incremental Enough for Continual Learning?](https://arxiv.org/abs/2112.02925)
- 会议：2021 (Arxiv)
- 作者：Andrea Cossu*, Gabriele Graffieti, Lorenzo Pellegrini, Davide Maltoni, Davide Bacciu, Antonio Carta, Vincenzo Lomonaco
- 主讲人：W

### [Does Continual Learning = Catastrophic Forgetting?]()
- 会议：2021 (Arxiv)
- 作者：Anh Thai, Stefan Stojanov, Zixuan Huang, Isaac Rehg, James M. Rehg
- 主讲人：W
  

## 2021-12-03

### [Memory Efficient Class-Incremental Learning for Image Classification](https://arxiv.org/abs/2008.01411)
- 会议：IEEE TNNLS 2021
- 作者：Hanbin Zhao, Hui Wang, Yongjian Fu, Fei Wu, Xi Li (Zhejiang University, College of Computer Science, Shanghai Institute for Advanced Study)
- 主讲人：W


## 2021-11-26


### [IIRC: Incremental Implicitly-Refined Classification](https://openaccess.thecvf.com/content/CVPR2021/papers/Abdelsalam_IIRC_Incremental_Implicitly-Refined_Classification_CVPR_2021_paper.pdf)
- 会议：CVPR 2021
- 作者：Mohamed Abdelsalam1, Mojtaba Faramarzi, Sarath Chandar（Mila), Shagun Sodhani (Facebook)  
- 主讲人：Z
- 内容：提出了持续学习中出现不同粒度的类别，且相互关系未知。用多标签分类的指标作为评价标准，在几个经典算法上观察了实验效果，说明了粗细粒度的关系会影响分类效果。
  
### [HCV: Hierarchy-Consistency Verification for Incremental Implicitly-Refined Classification](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjjqPXcw6L6AhWvG6YKHSQSBPwQFnoECAoQAQ&url=https%3A%2F%2Fwww.bmvc2021-virtualconference.com%2Fassets%2Fpapers%2F0008.pdf&usg=AOvVaw3EnyC0xyeNGIN5FUribl3I)
- 会议：BMVC 2021
- 作者：Kai Wang, Luis Herranz, Joost van de Weijer(University Autonoma de Barcelona)， Xialei Liu (Nankai University)
- 主讲人：Z
- 内容：针对IIRC问题提出了判断类别关系的方法，根据前面类别的打分确定是否为之前某一类的子类。超类和子类同时输出高分。测试时根据训练得到的层级关系调整矛盾的预测结果。


## 2021-11-19

### [Overcoming Catastrophic Forgetting in Incremental Few-Shot Learning by Finding Flat Minima](https://proceedings.neurips.cc/paper/2021/file/357cfba15668cc2e1e73111e09d54383-Paper.pdf)
- 会议：NIPS 2021
- 作者：Guangyuan Shi, Jiaxin Chen, Wenlong Zhang, Li-Ming Zhan, Xiao-Ming Wu
- 主讲人：H


### [Intriguing Properties of Contrastive Losses](https://proceedings.neurips.cc/paper/2021/file/628f16b29939d1b060af49f66ae0f7f8-Paper.pdf)
- 会议：NIPS 2021
- 作者：Ting Chen, Calvin Luo, Lala Li
- 主讲人：L

## 2021-11-12


### [Efficiently Identifying Task Groupings for  Multi-Task Learning](https://openreview.net/forum?id=hqDb8d65Vfh)
- 会议：NIPS 2021
- 作者：Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu,
Rohan Anil, Chelsea Finn (Google Brain)
- 主讲人：Z
- 内容：
  - 目的：设计高效的多任务分组方法。
  - 方法：提出inter-task affinity，用a任务的梯度方向观察b任务的损失函数变化情况，以此刻画任务间的相关程度。
  - 主要结论：此算法和SOTA相比在测试准确率不降低的情况下大幅减少了计算时间。


### [Meta-learning with an Adaptive Task Scheduler](https://openreview.net/forum?id=MTs2adH_Qq)
- 会议：NIPS 2021
- 作者：Huaxiu Yao, Yu Wang, Ying Wei, Peilin Zhao, Mehrdad Mahdavi, Defu Lian, Chelsea Finn
- 主讲人：H


## 2022-11-04

### [Can multi-label classification networks know what they don’t know?](https://openreview.net/forum?id=enKhMfthDFS)
- 会议：NIPS 2021
- 作者：Haoran Wang, Weitang Liu; Alex Bocchieri; Yixuan Li
- 主讲人：L
- 内容：启发于基于能量的OOD判别方法，本文针对多标签分类问题基于能量模型提出一种OOD鉴别指标。


## 2022-10-29


### [Few-shot Open-set Recognition by Transformation Consistency](https://openaccess.thecvf.com/content/CVPR2021/papers/Jeong_Few-Shot_Open-Set_Recognition_by_Transformation_Consistency_CVPR_2021_paper.pdf)
- 会议：CVPR 2021
- 作者：Minki Jeong, Seokeon Choi, Changick Kim (KAIST)
- 主讲人：L
- 内容：提出一种不需要训练集中包含未知样本的小样本开放集识别的方法。这种方法基于一类通过对类别原型进行变换的小样本识别方法，利用这种变换的一致性，通过取代原型的方法比较取代前后的距离来判断是否是unseen样本。

### [A continual learning survey: Defying forgetting in classification tasks](https://arxiv.org/abs/1909.08383)
- 会议：IEEE TPAMI 2021
- 作者：Matthias Delange (KU Leuven), Rahaf Aljundi (KU Leuven), Marc Masana (Computer Vision Center, Barcelona), Sarah Parisot (Huawei Noah’s Ark Lab, London), Xu Jia (Huawei Noah’s Ark Lab, Beijing),
Ales Leonardis (Huawei Noah’s Ark Lab), Greg Slabaugh (Huawei Technologies R&D, London), Tinne Tuytelaars (KU Leuven)
- 主讲人：W


## 2021-10-22

### [Co2L：Contrastive Continual Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Cha_Co2L_Contrastive_Continual_Learning_ICCV_2021_paper.pdf)
- 会议：ICCV 2021
- 作者：Hyuntak Cha, Jaeho Lee, Jinwoo Shin (KAIST)
- 主讲人：Z
- 内容：
  - 目的：将对比学习用于持续学习中。
  - 方法：1.将每个task的交叉熵损失换成监督型对比损失，并提出非对称损失，防止进一步分离之前见过的类（否则会出现存储样本与整体分布的偏差）。2.用instancewise relation distillation防止遗忘。
  - 主要结论：对比损失能提取更适合在任务间迁移的特征。

### [Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks](https://openreview.net/pdf?id=OdklztJBBYH)
- 会议：NIPS 2021
- 作者：Hanxun Huang, Yisen Wang, Sarah Erfani, Quanquan Gu, James Bailey, Xingjun Ma
- 主讲人：H
- 内容：在 WRN32-10 的下框架，探究怎样的结构有助于提高神经网络的对抗鲁棒性，发现当越靠近输出层宽度越小时，网络的对抗鲁棒性越强，当越远离输出层的网络参数越大时，对抗鲁棒性越强。

## 2022-10-08

### [Few-Shot Learning with Part Discovery and Augmentation from Unlabeled Images](https://www.ijcai.org/proceedings/2021/0313.pdf)
- 会议：IJCAI 2021
- 作者：Wentao Chen, Chenyang Si, Wei Wang, Liang Wang, Zilei Wang, Tieniu Tan (中科大，中科院计算所)
- 主讲人：L
- 内容：本文解决的小样本问题场景为：大量的无标签数据可作为特征提取器的预训练数据集。核心思想就是图像中关键Part的获取。在预训练部分，选择每张图像中信息量最大的Part，利用这一Part参与对比学习从而得到下游任务需要的特征提取器。在下游任务中，先用小样本数据训练一个分类器，再用该分类器对无标签数据进行预分类，选择分类概率较高的样本作为增强数据。增强的方法即通过一个样本的attention block，意在放大与类别相关的特征。最后再用增强的数据及原小样本数据重新训练分类器。

### [Data Augmentation for Meta-Learning](https://proceedings.mlr.press/v139/ni21a.html)
- 会议：ICML 2021
- 作者：Renkun Ni, Micah Goldblum, Amr Sharaf, Kezhi Kong, Tom Goldstein
- 主讲人：H
- 内容：本文探索了在元学习的不同阶段做数据增广，观察在每个阶段做数据增广的不同表现，得到数据增广在元学习不同阶段所能起到的不同作用



## 2022-09-17

### [Continual Learning in the Teacher-Student Setup: Impact of Task Similarity]()
- 会议：ICML 2021
- 作者：Sebastian Lee(Imperial College, London), Sebastian Goldt (SISSA), Andrew Saxe (University of Oxford)
- 主讲人：Z
- 内容：
  - 目的：实验观察持续学习中前后任务的相似性对结果的影响。仅涉及两层神经网络两个任务的情况。
  - 方法：teacher-student setup，ODE数值模拟。
  - 主要结论：intermediate task similarity leads to greatest forgetting


### 持续且无遗忘的深度学习方法研究
- 会议：博士学位论文
- 作者：胡文鹏
- 主讲人：W
- 内容：
  - 目的：通过持续学习克服灾难性遗忘问题，从表面原因（训练样本分布不均衡）与根本原因（特征偏置）下手
  - 本次主要汇报第3部分：
    - 1.参数生成与模型自适应方法（PGMA）：一种持续学习算法，针对表面原因
    - 2.全面学习（HL）：解决单类别分类，用全面正则项（H-reg）实现，解决根本原因
    - 3.全面持续学习框架：一种持续学习算法，结合H-reg，引入参数迁移、后处理机制，解决根本原因


