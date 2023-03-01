---
title: 梯度操控法持续学习
date: 2022-10-08
categories: [科研]
tags: [论文笔记, 持续学习]
img_path: /assets/img/
math: true
---

本文介绍持续学习的**梯度操控法**。它是一种防遗忘机制的方法，是直接规定、操控训练的更新过程的，通过修改梯度的计算、梯度下降公式来实现。

# 论文信息

### [Orthogonal Gradient Descent for Continual Learning](https://proceedings.mlr.press/v108/farajtabar20a.html)

- 会议：AISTATS 2020
- 作者：DeepMind

### [Continual learning of context-dependent processing in neural networks](https://www.nature.com/articles/s42256-019-0080-x)

- 期刊：Nature Machine Intelligence 2019
- 作者：中科院自动化所，类脑智能研究中心
  
### [Gradient Projection Memory for Continual Learning](https://openreview.net/forum?id=3AOj0RCNC2)

- 会议：ICLR 2021
- 作者：普渡大学

### [Gradient Episodic Memory for Continual Learning](https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)

- 会议：NIPS 2017
- 作者：Facebook

### [Efficient Lifelong Learning with A-GEM](https://openreview.net/forum?id=Hkf2_sC5FX)

- 会议：ICLR 2019
- 作者：牛津大学，Facebook

### [Continual Learning with Recursive Gradient Optimization](https://openreview.net/forum?id=7YDLgf9_zgm)

- 会议：ICLR 2022 (Spotlight)
- 作者：清华大学计算机系

--------------------



# 正交梯度下降（OGD）

这是一种直接修正梯度的方法。在训练新任务时，让参数向着**垂直于旧任务更新方向**更新。即对任务 $$t$$，旧任务在更新时的梯度方向 会张成子空间 $$V_{t-1}$$，让新任务更新时的梯度方向限制在该空间的正交补空间 $$V_{t-1}^\perp$$ 中进行（不用担心子空间占满了参数空间导致没有正交补，因为参数空间往往是很大的）。

当然这个梯度方向不是任意 $$V_{t-1}^\perp$$ 上的方向，首先要契合训练新任务，所以最终是将新任务原始梯度（分类损失正常反向传播计算的梯度）$$g$$ **投影**到 $$V_{t-1}^\perp$$，使用这个**修正**后的梯度 $$\tilde{g}$$ 作新任务的梯度下降参数更新：

$$\theta \leftarrow \theta - \eta \tilde{g}$$

这个子空间是属于旧任务的，所有必须在记忆中记住。下面讨论如何表示旧任务梯度子空间，即需要记住什么内容。表示一个子空间，通常是用空间中的向量。子空间由**旧任务梯度**张成的（以任务 $$t-1$$ 为例）：$$\{\nabla_\theta L(f(\mathbf{x};\theta),y)\}_{(\mathbf{x},y)\in \mathcal{D}_{train}^{(t-1)}}$$（注意 $$\theta$$ 一直在变化，每个求导点 $$\theta$$ 都不一样；目标函数由于 $$\mathbf{x}$$ 不同也不同）。

我们不能直接记住这些 $$\nabla_\theta L(f(\mathbf{x};\theta),y)$$，而应转换为正交基 $$S$$（用 Gram-Schmidt 公式，参考线性代数），原因有二：

- 正交基是空间的代表，数量较少，可减少记忆量；
- 算正交投影的公式（参考线性代数）只能用正交基计算：

$$ \tilde{g} = g - \sum_{v\in \mathcal{M}} proj_v(g)$$


记忆如何积累：记忆 $$\mathcal{M}$$ 里记录了旧任务子空间的正交基，随任务数增加是不断扩充的，因为 Gram-Schmidt 公式是迭代的。

还需要考虑以下问题：

- 为减少计算量和存储量，可以选用$$\{\nabla_\theta L(f(\mathbf{x};\theta),y)\}_{(\mathbf{x},y)\in \mathcal{D}_{train^{(t-1)}}$$的一部分而不是全部，例如 $$\mathbf{x}$$ 选 $$\mathcal{D}_{train^{(t-1)$$ 的一部分；对 $$C$$ 分类问题，$$ L(f(\mathbf{x};\theta),y)=[\nabla f_1(\mathbf{x};\theta), \cdots,\nabla f_C(\mathbf{x}_C,\theta)] [a_1-y_1, \cdots, a_C- y_C]^T$$，由于 $$y$$ 总是一个 one-hot 向量（即 $$y_1,\cdots,y_C$$ 只有一个 1，其他全是 0），可以只要 $$y_c = 1$$ （ground truth label）的，即 $$L(f(\mathbf{x};\theta),y) = \nabla f_c(\mathbf{x}; \theta)(a_c - 1)$$，原论文中称为 OGD-GTL；
- 可以用固定的旧任务训练好的 $$\theta^{(t-1)}_\star$$ 来代替旧任务梯度中一直变化 $$\theta$$，这样就可以在旧任务训练结束后再统一计算记忆的梯度，而不用边训练边算。

调节防遗忘程度的超参数算是选用的旧梯度数量。



# 递归最小二乘法（RLS）

这也是一种修正梯度的方法，它是将变换矩阵 $$P^{(t-1)}$$ 作用在梯度上：

$$ \theta \leftarrow \theta - \eta P^{(t-1)} g$$

这个变换矩阵记录了旧任务的某种知识，而且能够随任务 $$t$$ 迭代地积累知识。

假设任务采用线性回归模型 $$f(\mathbf{x}) = \mathbf{w}^T\mathbf{x}$$，由回归分析知识可知，在平方损失下，使用最小二乘法可得到参数的最优解公式：

$$\mathbf{w}^{\star} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

这个结果可以看作汇聚了该任务的所有知识。本方法就利用此式构建变换矩阵。首先不能直接拿来当作 $$P^{(t-1)}$$，它是参数，乘在参数梯度 $$g$$（量纲与参数一致）上是没有意义的。应当取 $$(\mathbf{X}^T\mathbf{X})^{-1}$$，它是一个方形矩阵，维度等于输入维数，也等于参数维数，所以能作用在参数梯度 $$g$$ 上。（它在统计学上应该有某种意义，但我不太清楚，欢迎大佬指点）

众所周知，直接计算这个矩阵是非常麻烦的，主要是因为 $$\mathbf{X}$$ 包含样本数太多。有近似算法——**递归最小二乘（RLS）算法**可以让 $$\mathbf{X}$$ 中的数据一个一个地来，迭代地计算。设 $$\mathbf{X} = [\mathbf{x}_1,\cdots, \mathbf{x}_n]$$，前 $$i$$ 个记为 $$\mathbf{X}_i$$，$$(\mathbf{X}_i^T\mathbf{X}_i)^{-1}$$ 的近似 $$P_i$$ 迭代计算公式：

$$ k_i = \frac{P_{i-1}\mathbf{x}_i}{\alpha + \mathbf{x}_i^T P_{i-1} \mathbf{x}_i}$$

$$P_i = P_{i-1} - k_i  \mathbf{x}_i^T P(i-1)$$

$$P_0$$ 初始化为单位矩阵 $$I$$。（该算法的原理是线性代数中子矩阵、矩阵分块的计算，感兴趣可以自己推导试试，不再详述。）


该方法的一大优点是所需记忆固定的，即一个固定大小的矩阵 $$P$$，不会随任务量增长。

记忆如何积累：面对持续学习，这种迭代的计算方式起到的作用不仅是简化了计算，也让任务之间可以继承，即在计算完毕任务 $$t-1$$ 的 $$(\mathbf{X}^T\mathbf{X})^{-1}$$ 后，可以以此为下一个任务迭代的初值，继续积累下去。也就是说，$$P^{(t-1)}$$ 不是只记录了任务 $$t-1$$，而是记录了所有旧任务 $$1,\cdots, t-1$$。



还需要考虑以下问题：

- 模型不一定是线性模型：对于深度为 $$L$$ 的网络，可以看成 $$L$$ 个线性模型，每层 $$l$$ 都有独立的 $$P$$ 作用在该层的梯度上（**layerwise**），这些 $$P$$ 的更新也是独立的。注意迭代公式里只用到了数据的输入（不用标签），所以 $$\mathbf{X}$$ 用每层的输入即可；
- 数据不是一个一个来的，而是一个 batch 来的：可以简单取 batch 的平均，当作一个数据，当然也有其他更好的方法；
- 可以在 $$P^{(0)}$$ 初始化时引入调节防遗忘程度的超参数 $$\lambda$$：$$P_0 = \lambda I$$。


# GPM


与 OGD 类似，区别在构造旧任务梯度的方式不一样。它不是就地取材从旧任务实际使用的梯度出发构造子空间，而是从数据下手，提取旧任务数据的信息来构造。也设任务采用线性回归模型，旧任务数据为 $$\mathbf{X} \in \mathbb{R}^{N\times p}$$。提取数据矩阵信息的一大工具是**奇异值分解（SVD）**：

$$\mathbf{X}_{N\times p} = \mathbf{U}_{N\times N}\mathbf{\Sigma}_{N\times p}\mathbf{V}^T_{p\times p}$$

我们要的是与梯度量纲一致的量，观察可以发现是右奇异向量 $$\mathbf{V}^T = [\mathbf{v}_1,\cdots, \mathbf{v}_p]$$，它的维度与参数梯度一致，可以在参数空间中张成子空间。与 OGD 一样，在训练新任务时，让参数在该子空间的正交补中更新即可。


记忆如何积累：由于奇异向量本身是正交的，直接当作正交基存到记忆里即可。


还需要考虑以下问题：

- 对于深度为 $$L$$ 的网络，也需要 layerwise，每层 $$\mathbf{X}$$ 用该层的输入即可；
- 数据不是整个来的，而是分 batch 来的：那计算奇异值分解也分 batch 来即可；
- 为减少计算量和存储量，可以取奇异向量的一部分。$$\mathbf{V}^T$$ 天生就按重要程度（奇异值）排序了，按照一定的准则取前 $$k$$ 个即可。个数 $$k$$ 算是调节防遗忘程度超参数。



# GEM, A-GEM







# RGO


## 一、持续学习：加正则项法

持续学习在训练第 $$k$$ 个任务时，损失函数可以分成

$$ \mathcal{L}  = \mathcal{L}^{\text{FINETUNE}} + \mathcal{L}^{\text{REVIEW}} $$

$$\mathcal{L}^{\text{FINETUNE}}$$ 是分类损失：

$$L_{k}(\theta)=\frac{1}{n_{k}} \sum_{i=1}^{n_{k}} l\left(f\left(\theta ; k, x_{i}\right), y_{i}\right)$$

加正则项法中的持续学习方法中，$$\mathcal{L}^{\text{REVIEW}}$$ 基本都是针对以下式子做改进：

$$F_{k}(\theta)=\sum_{j=1}^{k-1} L_{j}(\theta) \approx \sum_{j=1}^{k-1}\left[L_{j}\left(\theta_{j}^{*}\right)+\frac{1}{2}\left(\theta-\theta_{j}^{*}\right)^{T} H_{j}\left(\theta-\theta_{j}^{*}\right)\right] $$


对于 $$k$$ 之前的每一项任务，$$L_j(\theta) (j < k)$$ 本是可以像 $$L_k(\theta)$$ 那样用数据写出来的，但是现在的情况是没有重演数据，因此无法具体写出 $$L_j(\theta) (j < k)$$。这里考虑将其在之前训练好的参数 $$\theta_j^\star$$ 处泰勒展开，忽略高阶项作近似（注意在目标函数在最优值处梯度为0，所以没有一阶项）：

$$L_{j}(\theta) \approx \sum_{j=1}^{k-1} L_{j}\left(\theta_{j}^{*}\right)+\frac{1}{2}\left(\theta-\theta_{j}^{*}\right)^{T} H_{j}\left(\theta-\theta_{j}^{*}\right) $$

妙处在于这个式子就是负责解决遗忘的正则项，是一个要优化的东西，因此大部头的 $$L_{j}\left(\theta_{j}^{*}\right)$$ 是一个常数可以扔掉了，只需优化 $$\frac{1}{2}\left(\theta-\theta_{j}^{*}\right)^{T} H_{j}\left(\theta-\theta_{j}^{*}\right)$$。其中，$$H_{j}:=\nabla^{2} L_{j}\left(\theta_{j}^{*}\right)$$，这个Hessian矩阵是一个与（重演）数据无关的常值矩阵，虽然也无法直接计算，但有很多方法可以近似计算，例如[EWC](https://www.pnas.org/doi/10.1073/pnas.1611835114) 用Laplace近似将其替换为Fisher信息量矩阵的对角，[SI](https://arxiv.org/abs/1703.04200) 采用了更深入的方法。本文本质上也是在这上面作文章。

## 二、RLL


本文的第一步简化，是将上面这个REVIEW损失 $$F_k(\theta)$$ 改为 RLL (Recursive Least Loss)：


$$ F_{k}^{R L L}(\theta):=\frac{1}{2}\left(\theta-\theta_{k-1}^{*}\right)^{T}      \bar{H}  \left(\theta-\theta_{k-1}^{*}\right) $$

其中 $$ \bar{H} = \sum_{j=1}^{k-1} H_j$$。区别在于只需要存上个任务 $$k-1$$ 训练好的参数 $$\theta_{k-1}^\star$$，如果模型比较大、任务比较多，可以省下很多空间。


损失函数都要最终落实到梯度下降法等优化算法上，一般来说，定义好损失函数后，算法所需的梯度自然不必多解释。但本文采取了一种逆向思路：可以看到，目前的损失函数是 $$ F_k(\theta) + \lambda \cdot F_{k}^{R L L}(\theta)$$，作者不打算继续近似或修改这个损失函数了，想直接去考虑该如何梯度下降。


## 三、梯度的变换



如果不是持续学习，就是单纯地沿着分类损失 $$L_k(\theta)$$ 的负梯度进行梯度下降更新：

$$ \theta_{i}=\theta_{i-1}-\eta_{i} \nabla L_{k}\left(\theta_{i-1}\right)$$

持续学习加了REVIEW正则项 $$F_{k}^{R L L}(\theta)$$ 后，就是沿另一个方向梯度下降了。作者把这个方向归结为对$$L_k(\theta)$$ 的负梯度的线性变换

$$ \theta_{i}=\theta_{i-1}-\eta_{i} P \nabla L_{k}\left(\theta_{i-1}\right)$$

正则项 $$F_{k}^{R L L}(\theta)$$ 的形式与这个变换 $$P$$ 是有直接决定关系的。（当然，初始化的位置也与正则项绑定，例如L2正则项就相当于正态初始化。）

### REVIEW正则项与变换的不等关系

文章中的定理2给出了正则项与变换 $$P$$ 的关系，但是是一个不等关系：

$$ F_{k}^{R L L}\left(\theta_{k}^{*}\right) \leq \frac{1}{2} n_{k} \eta_{m} \hat{\sigma}_{m}(P \bar{H}) L_{k}\left(\theta_{k-1}^{*}\right) $$




可见如果取 $$P$$ 使得 $$\hat{\sigma}_{m}(P \bar{H})$$ 小，则对应的正则项（彻底训练后） $$ F_{k}^{R L L}\left(\theta_{k}^{*}\right)$$ 也大不到哪里去（不超过上式的上界）。

另外，一个与主题无关的事情，这个上界也与初始化有关系（上面都是假设初始化在 $$\theta_{k-1}^\star$$ ），初始化的位置若使 $$L_k$$ 较小，则 $$ F_{k}^{R L L}\left(\theta_{k}^{*}\right)$$ 也较小。<font color='green'>这就有问题了：$\mathcal{L}^{\text{FINETUNE}}$ 和 $\mathcal{L}^{\text{REVIEW}}$ 彼此是独立的，怎么会有关系？ </font>


### 收敛速度的一致性

这个一致性是对不同任务的一致性，收敛速度由学习率调节的。文章中的定理1说：

![T1](RGO_theorem1.png)

没有看懂，总之意思是如果 $$P$$ 是 $$\text{trace}(P) = \text{dim}(P)$$ 这样的矩阵话，一般来说所有任务收敛速率是一致的，只需给一个学习率即可。

调参问题是持续学习的一个痛点：怎么去调每个任务的超参数？调参是需要验证集的，持续学习的目标考虑的是在所有任务的性能，因此在为某个任务调参时，验证集必须包含前面的所有任务。如果：
- 来一个新任务就现场调一次参：首先调参的过程用到前面的数据（验证集是一个完整的数据，不像重演数据），就不是持续学习了；也违背机器学习的自动化理念（总不能让一个调参师守着这个模型，新任务一来就提醒他要干活了吧）；
- 从一开始就指定好各任务的超参数：首先要训几个任务是不知道的，如果定死了任务数量，就违反了“持续”的逻辑；第二，这样超参数的搜索空间太大了（上一种方案实际上是贪心的，此方案是穷举的），调也调不好。

如果像本文这样各任务只需指定一套超参数，就不会出现上面任何一个麻烦（其实此时一整套持续学习过程就打包成了整体——训练和测试是完全分离进行的，就调参而言和普通的学习没什么区别）。所以如果本文能真正把这件事从理论上搞定，那价值是很高的。


### 理想的变换

综上，要找的 $$P$$ 是这样的：

$$P:\left\{\begin{array}{l}
\min _{P} \hat{\sigma}_{m}(P \bar{H}) \\
\text { s.t. } \operatorname{trace}(P)=\operatorname{dim}(P)
\end{array}\right.$$

文章给出了此问题的解析解：

$$P=\frac{\operatorname{dim}(\bar{H})}{\operatorname{trace}\left(\bar{H}^{-1}\right)} \bar{H}^{-1}$$

问题来到了 $$\bar{H}$$ 的计算上。











## 四、$$\bar{H}$$ 计算的推导

本文硬算了这个 $$\bar{H}$$，但可不是每次都把 $$H_1, \cdots, H_{k-1}$$ 一个个算出来，而有一种巧妙的方式。

### 按层计算

为减少计算复杂度，本文采用按层的方式，即计算各层的 $$\bar{H}_l$$ （即 $$\bar{H}$$ 的子矩阵），构造 $$P_l$$：

$$ P_{l}=\frac{\operatorname{dim}\left(\bar{H}_{l}\right)}{\operatorname{trace}\left(\bar{H}_{l}^{-1}\right)} \bar{H}_{l}^{-1}$$

梯度下降也是按层更新参数。在文章中定理3保证了这样做在最优性上等同于整块网络一起处理，不再详述。

### $$\bar{H}_l$$ 的累加计算

记第 $$l$$ 层的参数为 $$h_l$$，则 $$\bar{H}_l = \sum_{j=1}^{k-1} H_{j,l} =\sum_{j=1}^{k-1} \frac{\partial^{2} L_{j}}{\partial h_{l}^{2}}$$。损失函数对参数的一阶导数我们是会算的，即链式法则：$$\frac{\partial^ L_{j}}{\partial h_{l} }= \frac{\partial L_{k}}{\partial h_{L}\frac{\partial h_{L}}{\partial h_{l}}$$，二阶导数（即Hessian矩阵）公式为（参考多元微积分）

$$ \frac{\partial^{2} L_{j}}{\partial h_{l} \partial h_{l}}=\left(\frac{\partial h_{L}}{\partial h_{l}}\right)^{T} \frac{\partial^{2} L_{j}}{\left(\partial h_{L}\right)^{2}}\left(\frac{\partial h_{L}}{\partial h_{l}}\right)$$


其中 $$\frac{\partial^{2} L_{j}}{\left(\partial h_{L}\right)^{2}} = \sum_{i=1}^{n_{j}}l^{\prime \prime}(f(\theta ; x_i) , y_i)$$。  最终形式为：

$$ \bar{H}_{l}=\sum_{j=1}^{k-1}  \sum_{i=1}^{n_j}\left(\frac{\partial h_{L}}{\partial h_{l}}\right)^{T} l^{\prime \prime}(f(\theta ; x_{i}) , y_{i}) \frac{\partial h_{L}}{\partial h_{l}}+\alpha I_{l}  $$

最后多加了 $$\alpha I_{l}$$ 为了保持矩阵的正定性（因为涉及它的逆）。鉴于此形式，完全可以存下任务 $$k$$ 计算的 $$\bar{H}_{l}$$，在下一次任务 $$k+1$$ 需要 $$\bar{H}_{l}$$ 时，累加一个 $$\sum_{i=1}^{n_{k}}l^{\prime \prime}(f(\theta ; x_i) , y_i)$$ 即可。

我们需要的 $$\bar{H}_l$$ 包含了之前所有任务知识的精华，正是由于它的**可累加性**，所以可以每次新来任务时，只使用现在来的数据即可轻松地计算。这也是本文的思想精髓。



## 算法框架


### RLS算法

目前为止，各部分已定义清楚，已经可以算法实现了：新任务 $$k$$ 来临时，分层计算 $$\bar{H}_l$$，再代入公式得到变换 $$P_l$$，最后作变换的梯度下降。但是求 $$\bar{H}_l$$ 的逆是非常复杂的（ $$O(n^3)$$ 复杂度），实际硬算不可取。

作者使用了递归最小二乘（RLS, Recursive Least Square) 算法解决矩阵求逆的问题，文章没有细说，我愣是没有搞明白。以下是最终的算法框架：


![算法](RGO_algorithm.png) 

可见没有涉及计算 $$\bar{H}_l$$，求其逆的过程被中间涉及 $$k_l$$ 的几步巧妙地转化了。






### FEL机制

本文还应用了一个逻辑上相对独立的机制，称为FEL（Feature Encoding Layer）。这个机制是在进行新任务的训练/测试时，将网络每层的神经元按照固定的方式重排一下。好处是很容易想象的，可以避免网络以偷懒的方式死记硬背当前任务的知识，因为下个任务即被打乱，则迫使各个神经元学到更加 general 的知识。我认为这也是一种防止遗忘的方式。

文章以 $$S_l(k)$$ 记在第 $$k$$ 个任务时第 $$l (l=1,2,\cdots,L)$$ 层的随机重排。应注意这里写成了 $$k$$ 的函数形式，即意味着固定，可理解为随机以 $$(k)$$ 作为随机数种子。

此机制并没有在上面的算法框架中得到体现。


## 总结

实验就不说了，SOTA就完事了。这篇文章主要还是理论上很有深度，和很多持续学习看上去比较naive的方法不一样，里面的几个小点：FEL、收敛速度一致性都很有借鉴意义，值得深入考虑一下。





