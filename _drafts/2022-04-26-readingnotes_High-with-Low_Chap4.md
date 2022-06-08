---
title: 读书笔记：《低维模型下的高维数据分析》第4章：低秩矩阵恢复
date: 2022-04-26
categories: [科研]
tags: [读书笔记, 《低维模型下的高维数据分析》]
img_path: /assets/img/
math: true
---


## 书籍信息 



### [High-Dimensional Data Analysis with Low-Dimensional Models: Principles, Computation, and Applications](https://book-wright-ma.github.io)

- 作者：
    - [John Wright](http://www.columbia.edu/~jw2966/) - 哥伦比亚大学
    - [马毅](https://people.eecs.berkeley.edu/~yima/) - UC Berkeley


------------------------------

## 低秩矩阵问题

将上两章的向量重建问题推广为**矩阵重建问题**：

$$ \mathcal{A} \[\mathbf{X} \text{(unknown)}\]  = \mathbf{y} \text{(observation)} $$

其中 $$\mathbf{x}$$ 变成了矩阵：$$\mathbf{x} \in \mathbb{R}^{n_1\times n_2}$$；观测 $$\mathbf{y}$$ 既可以是向量，也可以推广为矩阵（这里是向量），仍为 $$m$$ 个：$$\mathbf{y} \in \mathbb{R}^m$$； $$\mathbf{A}$$变成一个将矩阵映射为向量的线性映射：$$ \mathbf{A}: \mathbb{R}^{n_1\times n_2}\rightarrow \mathbb{R}^m$$。重建问题的目标仍是：已知观测 $$\mathbf{y}$$ 与生成映射 $$\mathbf{A}$$，恢复出未知的 $$\mathbf{X}$$。

同样地，当观测数据不够，即 $$m < n_1\times n_2$$ 时，这个重建问题是病态的（ill-posed），因为此方程解不唯一。为了使问题 well-posed，对 $$\mathbf{X}$$ 现在是作**低秩性假设**：$$\mathbf{X}$$ 秩应当足够低。向量重建问题变为**低秩矩阵恢复问题**，形式化为一个矩阵秩优化问题，称为 Affine Rank Minimization：

$$ \begin{array}{lc}
\min & \operatorname{rank}(\mathbf{X}), \\
\text { subject to } & \mathcal{A}\[\mathbf{X}\]=\mathbf{Y}
\end{array} $$

本章的理论就讨论这件事情。


解决上述问题的基础是**低秩矩阵近似问题**：寻找一个矩阵 $$\mathbf{Y} \in \mathbb{R}^{n_1\times n_2}$$ 的低秩矩阵近似 $$\mathbf{X}\in \mathbb{R}^{n_1\times n_2}$$，形式化为

$$\begin{array}{ll}
\min & \|\boldsymbol{X}-\boldsymbol{Y}\|_{F}, \\
\text { subject to } & \operatorname{rank}(\boldsymbol{X}) \leq r
\end{array}$$

它也可以写作矩阵秩优化问题：

$$\begin{array}{ll}
\min & \operatorname{rank}(\boldsymbol{X}), \\
\text { subject to } & \|\boldsymbol{X}-\boldsymbol{Y}\|_{F} \leq \varepsilon .
\end{array}$$






以下是几个例子（节选书上与 AI 相关的）：

### 推荐系统中的协同过滤

低秩矩阵恢复问题的一个应用是**矩阵补全问题**：一个矩阵有些位置的值是缺失的，想猜测它们的值，把整个矩阵补全。如果对矩阵没有任何限制，则填什么值都可以，问题是病态的；而对矩阵作限制后则缩小了可填值的范围，问题可能转为 well-posed。

形式化如下：已知的是一个不完整的矩阵 $$\mathbf{Y}$$，想要补全为完整的矩阵 $$\mathbf{X}$$，设 $$\Omega \doteq\{(i, j) \mid \text { user } i \text { has rated product } j\}$$。优化问题为：


$$ \begin{array}{lc}
\min & \operatorname{rank}(\mathbf{X}), \\
\text { subject to } & \mathcal{P}_{\Omega}\[\mathbf{X}\]=\mathbf{Y}
\end{array} $$

其中 $$\mathcal{P}_{\Omega}$$ 为在 $$\Omega$$ 子空间上的投影映射，线性代数的知识告诉我们这是一个线性映射。

$$ \mathcal{P}_{\Omega}[\boldsymbol{X}](i, j)= \begin{cases}X_{i j} & (i, j) \in \Omega \\ 0 & \text { else }\end{cases} $$



![CF](collaborative_filtering.png)


推荐系统的任务就是根据有相似行为的用户推荐相似的内容，这个技术叫协同过滤（Collaborative Filtering），它的原理就是在解矩阵补全问题。

推荐系统考虑的矩阵是一个评分矩阵，每行表示一个客户，各列表示用户对各物品的评分。对评分矩阵的限制就是**低秩**，这与“相似的用户喜欢相似的内容”一致。

### NLP中的潜在语义分析

潜在语义分析的目标是从一系列文档中提取话题进行分析，是一个低秩矩阵近似问题。它考虑的矩阵是词汇-文章矩阵 $$ \mathbf{Y} = [\mathbf{y}_1 | \cdots |\mathbf{y}_{n_2} |]$$，每列向量表示各文档中不同词语出现的频率。单独拿出一篇文章看，我们希望它可以近似表示成一些话题 $$\mathbf{t}_1, \cdots, \mathbf{t}_l$$ 的复合：

$$ \mathbf{y}_{n_j} \approx \sum_{l=1}^r \mathbf{t}_l \alpha_{l,j}$$

其中 $$\alpha$$ 是权重，又称 abundance。将文章拼起来看，就是 

$$ \mathbf{Y} \approx \mathbf{T} \mathbf{A}$$

要做的是从 $$\mathbf{Y}$$ 中尽量地概括话题，即求其低秩近似，并分解为 $$\mathbf{T}$$ 和 $$\mathbf{A}$$（低秩保证了分解出的 $$\mathbf{T}$$ 中包含足够少的话题）。




## 数学工具

和上一章一样，为建立理论，要建立起对操作对象矩阵的稀疏性的衡量标准，也要对 $$ \mathcal{A}$$ 的线性无关性作衡量。除此之外，本章还需用到奇异值分解这个重要工具。



### 奇异值分解

**奇异值分解**（SVD）是本章要用的重要工具，不再详述此理论，仅列出结论：

任何矩阵 $$ \mathbf{X} \in \mathbb{R}^{n_1\times n_2}$$都可以分解成

$$ \mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\star$$

的形式，其中 $$\mathbf{\Sigma}$$ 是将奇异值 $$\sigma_i$$ 按大小顺序放在对角线上的矩阵，$$\mathbf{U},\mathbf{V}$$ 为维数为 $$n_1, n_2$$ 的正交矩阵。它们包含了奇异向量 $$\mathbf{u}_i, \mathbf{v}_i$$，SVD 又可写成

$$\mathbf{X}=\sum_{i=1}^{\min \left(n_{1}, n_{2}\right)} \sigma_{i} \boldsymbol{u}_{i} \boldsymbol{v}_{i}^{*}$$

求矩阵的 SVD 是一个非凸问题，但可以全局、高效地解出来，例如幂迭代（Power Iteration）算法。作者花了大篇幅讲述这个问题，感兴趣可以参考原书。



#### SVD解决低秩矩阵近似问题

这是奇异值分解的一个应用例子。
考虑上面低秩矩阵近似的优化问题，设原矩阵 SVD 分解为 

$$\mathbf{Y}=\sum_{i=1}^{\min \left(n_{1}, n_{2}\right)} \sigma_{i} \boldsymbol{u}_{i} \boldsymbol{v}_{i}^{*}$$

可以证明，这个优化问题的解为 

$$\hat{\boldsymbol{X}}=\sum_{i=1}^{r} \sigma_{i} \boldsymbol{u}_{i} \boldsymbol{v}_{i}^{*}$$

其中 $$r$$ 为约束条件里约束的秩。因此解低秩矩阵近似只需奇异值分解，然后将其截断到要求的秩即可。



### 矩阵稀疏性的衡量

衡量矩阵的稀疏性除了秩，还可以用**核范数**（nuclear norm）衡量：

$$\|\boldsymbol{X}\|_{*}=\sum_{i} \sigma_{i}(\boldsymbol{X})$$

即矩阵的所有奇异值之和，它可以看作秩的推广：记奇异值组成向量 $$\sigma(\mathbf{X})=(\sigma_1, \cdots, \sigma_{\min{n_1,n_2}})$$， 则秩是 $$\|\sigma(\mathbf{X})\|_0$$，核范数是  $$\|\sigma(\mathbf{X})\|_1$$。

核范数的一些性质列举如下，不再详细讨论：
- 核范数是范数；
- 




### $$\mathcal{A}$$ 线性无关性的衡量标准




## 恢复理论


### 恢复定理：



### 松弛：到L0

低秩矩阵恢复的 Affine Rank Minimization 问题可以划归为上一章的 L0 Minimization 问题：


反之亦然，


### 推广到核范数


### 恢复定理：
