---
title: 论文笔记：Continual Learning with Recursive Gradient Optimization
author: Shawn Wang
date: 2022-04-18
categories: [科研]
tags: [论文笔记, 持续学习]
math: true
---


## 论文信息 



### [Continual Learning with Recursive Gradient Optimization](https://openreview.net/forum?id=7YDLgf9_zgm)


- 会议：ICLR 2022 (Spotlight)
- 作者：
    - Hao Liu - 清华大学计算机系（可能是后者的学生）
    - [刘华平](https://www.cs.tsinghua.edu.cn/info/1122/3566.htm) - 清华大学计算机系
- 内容：本文可以看成是加正则项的持续学习方法。


------------------------------



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

![T1](/assets/img/2022-04-18_2.png)

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


![算法](/assets/img/2022-04-18_1.png) 

可见没有涉及计算 $$\bar{H}_l$$，求其逆的过程被中间涉及 $$k_l$$ 的几步巧妙地转化了。






### FEL机制

本文还应用了一个逻辑上相对独立的机制，称为FEL（Feature Encoding Layer）。这个机制是在进行新任务的训练/测试时，将网络每层的神经元按照固定的方式重排一下。好处是很容易想象的，可以避免网络以偷懒的方式死记硬背当前任务的知识，因为下个任务即被打乱，则迫使各个神经元学到更加 general 的知识。我认为这也是一种防止遗忘的方式。

文章以 $$S_l(k)$$ 记在第 $$k$$ 个任务时第 $$l (l=1,2,\cdots,L)$$ 层的随机重排。应注意这里写成了 $$k$$ 的函数形式，即意味着固定，可理解为随机以 $$(k)$$ 作为随机数种子。

此机制并没有在上面的算法框架中得到体现。


## 总结

实验就不说了，SOTA就完事了。这篇文章主要还是理论上很有深度，和很多持续学习看上去比较naive的方法不一样，里面的几个小点：FEL、收敛速度一致性都很有借鉴意义，值得深入考虑一下。





