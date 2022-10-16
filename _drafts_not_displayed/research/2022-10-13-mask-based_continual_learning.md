---
title: 基于 mask 机制的持续学习
date: 2022-10-13
categories: [科研]
tags: [学习笔记, 持续学习]
img_path: /assets/img/
math: true
---

本文详细讲解在持续学习中加 mask 机制的论文。Mask 机制的基本逻辑与分类见笔记[《持续学习基础知识》]()，本文只讲解每篇论文如何对应到我在该笔记中作的分类。

注意，基于 mask 机制的持续学习天然要求场景为 TIL，即要求给出数据的任务 ID $$t$$。

## 论文信息


### [Ternary Feature Masks: zero-forgetting for task-incremental learning](https://openreview.net/forum?id=oLvlPJheCD)

- 期刊：TMLR 2022
- 作者：Tianlong Chen, Sijia Liu, Shiyu Chang, Lisa Amini, Zhangyang Wang




# PackNet

论文链接：[PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf), CVPR 2018


最简单的加 mask 方法是硬隔离：每来一个新任务，先按照固定算法分配它的 mask，mask 实现为二元的 weight mask，mask 之间不重叠；训练新任务时，前向传播、反向传播都完全遮住旧任务。但这样与独立式学习没有什么区别。

为了避独立式学习之嫌，PackNet 论文里采用的方式是：来新任务时，**先将可用参数全部训练，再剪掉（prune）一部分参数预留给后面任务使用**，剩下的留给当前任务。剪掉参数的行为势必会引起效果的断崖式下跌，需要**重新训练**（re-train）保留下来的参数，但训练力度就不必像之前正式训练那样了，可以少些 epoch。

在学习后面任务时，将前面的任务。。过程如下图，注意该图每个圈代表一个权重，这可能表示的是中间某全连接层（5维到5维）的权重。

![](PackNet_training.png)

剪参数是**固定算法**：文章的做法是直接保留参数若干个绝对值最大的，剪掉剩下的（剪掉的比例固定，作为超参数）。各任务 mask 是不重叠的，导致模型容量问题。论文通过将网络向外扩充来解决。

# Piggyback

论文链接：[Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.pdf), ECCV 2018

本文实现为二元的 weight mask，并且可学习。解决不可导的方式是先将 mask $$\mathbf{M}_l$$ 看成实值的 $$\mathbf{M}_l^r$$ 来学习，再用固定的阈值函数转换为二元 mask：

$$\mathbf{M}_l^r = \mathbf{1}_{>\tau}(\mathbf{M}_l)$$

阈值函数 $$\mathbf{1}_{>0}(\cdot)$$ 是指一个二值函数，大于 $$\tau$$ 取 1，否则取 0，$$\tau$$ 为超参数。注意，学完了 $$\mathbf{M}_l^r$$ 即可立即通过阈值函数转换为 $$\mathbf{M}_l$$， $$\mathbf{M}_l^r$$ 只是中间变量，无须存储。


本文的另一重要特色是真正的**网络参数一直是固定的**，整个训练过程只学对任务做出网络选择的 mask，而网络本身作为一个参照物不会被更新。这个固定的网络称为 backbone，对于每个任务测试时，相当于 backbone 背上了（piggyback，就是英文中把人背在背上的意思）一个 mask，换另一个任务就卸下该 mask，再背上另一个 mask。可以，这样的模式一定会要求 backbone 的初始化要非常的好，作者也做了大量的实验验证 backbone 初始化带来的影响。

至于为什么可以固定 backbone 参数，可以参考 [SupSup]() 论文，总体的思想是：

1. 网络参数 $$N$$ 足够多时，二元 mask 的组合也是足够多的（$$2^N$$），仅背 mask 表示能力就很强了；
2. 通过实验发现，即使随机初始化 backbone 仅背 mask 也能达到不错的效果，更何况 backbone 大都使用大型网络如 VGG、ResNet 的预训练初始化，它们本身已经预训练得很好了。


在本方法中，每个任务都是完全独立地学 mask，因此学到的 mask 是有可能重叠的。另外，backbone 网络是固定的，学习过程也就不会倾向于学到全 1 的 mask 了，不需设计稀疏化机制。个人认为此工作有很大的独立式学习嫌疑。


# HAT

论文链接：[Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://proceedings.mlr.press/v80/serra18a.html)，ICML 2018

本文实现为二元的 feature mask，并且可学习，没有固定 backbone 参数。解决不可导的方式也是先将 mask $$\mathbf{m}_l$$ 按照如下公式转换为对实值的 $$\mathbf{e}_l$$（沿用论文中记号，称为 task embedding）学习：

$$ \mathbf{m}_l = \mathbf{1}_{>0}(\mathbf{e}_l)$$

$$\mathbf{e}_l$$ 也算是中间变量，无须存储。

> 注意它与 Piggyback 中 $$\mathbf{M}_l^r$$ 重要区别是反向传播时 Piggyback 不会涉及阈值函数，而 HAT 会涉及。
{: .prompt-warning }

此时阈值函数 $$ \mathbf{1}_{>0}$$ 仍然不可导。解决思路是，由于它可以看作 S 型函数 $$S(\cdot) = \sigma(s \cdot)$$ 在 $$s\rightarrow +\infty$$ 的极限，在训练时使用光滑的 $$S(\cdot)$$ 去近似它（但测试时要使用 $$\mathbf{1}_{>0}）。但不能直接取一个很大的 $$s$$ 就完事了，这里用到的策略是**退火**（annealing），在训练过程中让每个 batch 取的 $$s$$ 越来越大：

$$s = \frac1{s_{max}} + (s_{max} - \frac1{s_{max}}) \frac{b-1}{B-1}, b = 1,\cdots, B$$

$$s_{max}$$ 是预先设定好的很大的数，据此公式，最后一个 batch 取的 $$s$$ 就是 $$s_{max}$$。

它采用的**稀疏化机制**是在损失函数上加一个正则项：

$$ R(\mathbf{m}^{(t)}, \mathbf{m}^{(\leq t-1)})= \frac{\sum_{l=1}^{L-1}\sum_{i=1}^{N_l  } m_{l,i}^{(t)}(1- m_{l,i}^{(\leq t-1)}}{\sum_{l=1}^{L-1}\sum_{i=1}^{N_l} (1- m_{l,i}^{(\leq t-1)}}$$

注意该函数的自变量是 $$\mathbf{e}^{(t)}$$，包含在 $$\mathbf{m}^{(t)}$$ 中。这个正则项的意思是，在旧 mask 为 0 的位置，新 mask 为 1 的尽量地少（为 0 的尽量地多），尽量使得新 mask 与旧 mask 重合。对于任务 1，可定 $$\mathbf{m}^{(0)}$$ 为全 0 mask，可直接限制 $$\mathbf{m}^{(1)}$$ 的稀疏程度。

训练任务 $$t$$ 的前向与反向传播过程如下（Compensation 是文章额外引入的机制，不打算讲解）：

![HAT_forward_and_backward_passes](HAT_forward_and_backward_passes.png)

注意，训练新任务时并没有实际使用旧任务的 mask 进行前向传播，而只是用在了稀疏正则项上（$$\mathbf{m}^{(\leq t-1)}$$）。


# APD

论文链接：[Scalable and Order-robust Continual Learning with Additive Parameter Decomposition](https://openreview.net/pdf?id=r1gdj2EKPB), ICLR 2020



# SupSup

论文链接：[Supermasks in Superposition](https://proceedings.neurips.cc/paper/2020/file/ad1f8bb9b51f023cdc80cf94bb615aa9-Paper.pdf), NIPS 2020

本文实际上是把 Piggyback 那种固定 backbone 参数的方法推广到了更多的持续学习场景：

- 训练数据与测试数据都有任务 ID 信息（论文中简称 GG）：TIL 场景，即 Piggyback；
- 训练数据有任务 ID 信息，测试数据没有（论文中简称 GN）；
- 训练数据与测试数据都没有任务 ID 信息（论文中简称 NN）：不打算讲解。

对于 GN 场景，它与 TIL 场景区别在测试阶段有无任务 ID 信息。本文的算法试图先预测出任务 ID，再按照 TIL 处理，而不是一步搞定。对于固定 backbone 的 mask 机制，测试时的输出：

$$\tilde{f}(\mathbf{x}; \mathbf{W}, \mathbf{\alpha}) = f(\mathbf{x}; \mathbf{W} \odot (\sum_{t=1}^T \alpha_t \mathbf{M}^t)) $$

其中 $$f(\mathbf{x};\mathbf{W})$$ 是固定的 backbone，固定参数为 $$\mathbf{W}$$，$$\mathbf{\alpha}=[\alpha_1,\cdots,\alpha_C], \alpha_c \in [0,1]$$ 是根据测试数据 $$\mathbf{x}$$ 确定的一个预测任务 ID 的变量（最好是一个 one-hot 向量）。

如何根据测试数据 $$\mathbf{x}$$ 确定 $$\mathbf{\alpha}$$？作者的方法是将 $$\tilde{f}(\mathbf{x}; \mathbf{W}, \alpha)$$ 看作 $$\mathbf{\alpha}$$ 的分布 $$p(\mathbf{\alpha})$$，使得该分布的熵尽量地大。

发生在测试阶段，








# TFM

论文链接：[Ternary Feature Masks: zero-forgetting for task-incremental learning](https://openaccess.thecvf.com/content/CVPR2021W/CLVision/papers/Masana_Ternary_Feature_Masks_Zero-Forgetting_for_Task-Incremental_Learning_CVPRW_2021_paper.pdf), CVPR Workshop 2021




# KSM

论文链接：[KSM: Fast Multiple Task Adaption via Kernel-wise Soft Mask Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_KSM_Fast_Multiple_Task_Adaption_via_Kernel-Wise_Soft_Mask_Learning_CVPR_2021_paper.pdf), CVPR 2021




# CTR

论文链接：[Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning](https://proceedings.neurips.cc/paper/2021/file/bcd0049c35799cdf57d06eaf2eb3cff6-Paper.pdf), NIPS 2021