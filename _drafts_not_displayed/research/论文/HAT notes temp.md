
# HAT

论文链接：[Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://proceedings.mlr.press/v80/serra18a.html)，ICML 2018

本文实现为二元的 feature mask，并且可学习，没有固定 backbone 参数。解决不可导的方式也是先将 mask $\mathbf{m}_l$ 按照如下公式转换为对实值的 $\mathbf{e}_l$（沿用论文中记号，称为 task embedding）学习：

$$ \mathbf{m}_l = \mathbf{1}_{>0}(\mathbf{e}_l)$$

$\mathbf{e}_l$也算是中间变量，无须存储。

此时阈值函数 $ \mathbf{1}_{>0}$ 仍然不可导。解决思路是，由于它可以看作 S 型函数 $S(\cdot) = \sigma(s \cdot)$在 $s\rightarrow +\infty$ 的极限，在训练时使用光滑的 $S(\cdot)$ 去近似它（但测试时要使用 $\mathbf{1}_{>0}$）。但不能直接取一个很大的 $s$ 就完事了，这里用到的策略是**退火**（annealing），在训练过程中让每个 batch 取的 $s$ 越来越大：

$$s = \frac1{s_{max}} + (s_{max} - \frac1{s_{max}}) \frac{b-1}{B-1}, b = 1,\cdots, B$$

$s_{max}$ 是预先设定好的很大的数，据此公式，最后一个 batch 取的 $s$ 就是 $s_{max}$。

它采用的**稀疏化机制**是在损失函数上加一个正则项：

$$ R(\mathbf{m}^{(t)}, \mathbf{m}^{(\leq t-1)})= \frac{\sum_{l=1}^{L-1}\sum_{i=1}^{N_l  } m_{l,i}^{(t)}(1- m_{l,i}^{(\leq t-1)}}{\sum_{l=1}^{L-1}\sum_{i=1}^{N_l} (1- m_{l,i}^{(\leq t-1)}}$$

注意该函数的自变量是 $\mathbf{e}^{(t)}$，包含在 $\mathbf{m}^{(t)}$ 中。这个正则项的意思是，在旧 mask 为 0 的位置，新 mask 为 1 的尽量地少（为 0 的尽量地多），尽量使得新 mask 与旧 mask 重合。对于任务 1，可定 $\mathbf{m}^{(0)}$ 为全 0 mask，可直接限制 $\mathbf{m}^{(1)}$ 的稀疏程度。

注意，训练新任务时并没有实际使用旧任务的 mask 进行前向传播，而只是用在了稀疏正则项上（$\mathbf{m}^{(\leq t-1)}$）。但是，不允许更新旧任务 mask 住的参数，图中 $g_l$ 到 $g'_l$，这个需要在反向传播时手动将对应的梯度乘以旧任务 mask 的反转值。各个任务 mask 之间的联系仅此而已。
