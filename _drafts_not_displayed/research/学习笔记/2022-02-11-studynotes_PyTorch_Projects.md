---
title: PyTorch 学习笔记：工程性知识
date: 2022-02-11
categories: [科研]
tags: [读书笔记, 《动手学深度学习》, 机器学习, 技术]
img_path: /assets/img/
math: true
---

本文汇总使用 PyTorch 搭建项目时的一些边缘性的工程性知识，让代码真正地成为一个完整的深度学习项目。这部分内容包括如何可视化数据、读写训练进度等。本文参考 [Dive into Deep Learning (PyTorch 版)](https://d2l.ai) 中的以下内容：
- 5.5 节：读写文件；


 
------------------------------





在真正开始训练前，作double check 。cs231n



# 读写训练进度

深度学习程序的一个特点是运行时间长，一个任务经常需要跑几天、几个月。可以把深度学习程序比作 RPG 游戏，打通关时间长的 RPG 时我们需要定期存档，不仅为了下一次打开游戏时接续进度，还能预防电脑未响应、死机等突发情况导致游戏白打，甚至有时需要换台电脑玩这个进度、应当存档拷贝到新电脑；而且有时候会存多份档，为了预防游戏中某一次策略错误（如，买错了道具；打 boss 打不过去或者游戏有 bug 导致的陷入死循环，俗称坏档）导致的严重后果，起到后悔药的作用。

大型的深度学习程序需要**定期存档且存多份档**，和上面是一个道理，不必多解释了。它与游戏的不同在于用户无法在运行过程中手动控制，只有停止程序这一个选择；定期存档的操作需要预先写进代码里。

深度学习程序也是 Python 程序，当然可以用 Python 自带的文件读写功能，将变量保存于本地文件。但 PyTorch 为深度学习设计了专门更高级的 API，更加方便，最好使用这套 API。PyTorch 可以读写 Tensor 对象，`nn.Parameter` 对象，还可以是 `{字符串:Tensor或Parameter}` 的字典：
- `torch.save(obj, path)`：将对象 obj 存到路径为 path 的文件中；
- `obj = torch.load(filename)`：将 filename 文件存储的变量赋值到 obj。
此类文件属文本文件，PyTorch 推荐使用扩展名 `.pt`,`.pth`（书中使用了 `.params`）。存档文件最好存储在项目单独的一个子目录下。

深度学习最需要存档的东西是**模型参数**，它是训练的目标。网络结构无需保存，因为它就写在代码里，只需保存其参数即可。保存模型参数的推荐方法是存它的 `.state_dict()`（前面说过它是存所有模型参数的字典），因为 `nn.Module` 有一个方便的 API：`net.load_state_dict(state_dict)`，能将 state_dict 一步读取所有参数到模型 net 中。

除了模型参数，还有一些必须存档的信息：当前 epoch 轮数，优化器里还有一些状态量（`optimizer.state_dict()`），如果用了调度器它也有状态量 `scheduler.state_dict()`，等等。可以将其统统打包成一个字典，类似下面的做法：
```python
checkpoint = {
    'epoch': epoch,
    'net': net.state_dict(),
    'optimizer': optimizer.state_dict()
}
```
除此之外，为了方便，也可以打包进去其他需要记住的东西，如超参数、配置变量、当前 loss 等统计信息，等等。写到字典里是为了方便程序内使用，如果只是给人看一下，一些小的信息也可以传给 `path`，写到文件名内。

下面讨论存档的频率。首先要说一点，为了实现多份存档，文件名最好不一样，防止覆盖。存档太频会浪费硬盘空间，例如一个 batch 或 epoch 一存；太不频则有更大的重新训练风险。而且并不是所有的档都需要存，和游戏一个道理，一般是在比较关键的进度存一下档。常见的做法是在训练循环体中设置条件判断语句写的检查点（checkpoint），判断是不是关键的存档。

以下是一套完整的流程（引自[知乎](https://www.zhihu.com/question/340567722/answer/2505072802)，作者“”人类之奴）：
```python
start_epoch = -1

# 如果接续训练（RESUME=1），则加载 checkpoint
if RESUME:
    path_checkpoint = checkpoint_path
    checkpoint = torch.load(path_checkpoint)
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_schedule'])


for epoch in range(start_epoch+1, num_epochs):
    train(net, train_loader)
    test_loss = test(net, test_loader)

    # 检查点：测试集 loss 小于一定阈值。epoch 小于一半总训练轮数时认为训练不够，不设检查点
    min_loss_val = 1
    if epoch > int(num_epochs/2) and test_loss <= min_loss_val: 
        min_loss_val = test_loss
        checkpoint = {
            'loss':test_loss,
            'epoch':epoch,
            'net':net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'lr_schedule':scheduler.state_dict()}
        if not os.path.isdir(r'tf-logs/'+save_model):
            os.mkdir(r'tf-logs/'+save_model)
        torch.save(checkpoint,r'tf-logs/'+save_model+'/ckpt_best_%s.pth'%(str(epoch+1)))
```

> 读写功能除了上述存档接续训练进度，还有其他常见的应用场景：例如保存训练好的参数给别人使用。常见的大型网络可以使用别人预训练的权重，再在自己的任务上微调，这些预训练权重通常保存在 `.pth` 文件中，从网上下载。
{: .prompt-tip }



# 可视化


深度学习里很多内容需要**可视化**，辅助研究，例如：

- 数据集中的数据；
- 网络结构；
- 学习曲线、指标的变化曲线等。

为实现此目的，除了可以手动调用例如 Matplotlib 等天然的可视化工具，深度学习框架也开发了自己的可视化工具。本文介绍 TensorBoard 的使用，它是 TensorFlow 的可视化工具，目前也支持了 PyTorch（我使用后者，本文讲后者）。


# TensorBoard 逻辑

TensorBoard 的逻辑可以看成一个画家，以及一个画布，给画家各种作画指示，它就会按要求在画布上作出各种图。

工具的两个部分：
- **画家**和各种**作画指示**：在 `torch` 库中，存放在 `torch.utils.TensorBoard`。画家是 `TensorBoard.SummaryWriter` 类，作画指令就是类方法 `add_xx(...)`（xx 表示各种支持的内容，例如 scalar、graph 等），每调用一次就会在画布上画方法参数中对应的内容；
- **画布**：是一个本地软件，在本地端口运行（浏览器打开，类似于 Jupyter Notebook），需要额外安装。必须启动画布，才能看到画家作的画。

问题来了，画家和画布是两个程序，画布怎么知道画家的作画内容呢？这是通过**日志**实现的。画家作画其实是输出了一些画布能读懂的日志，画布通过输入日志来呈现画家的作画。这些日志存放在文本文件（称为**日志文件**），并通常放于专门的日志目录下（在代码中，画家和画布都是从指定目录下输出、输入日志），使用时应当为画家和画布指定**相同的日志目录**。

命令总结：

- 安装画布：`conda install tensorboard`；
- 启动画布：`tensorboard --logdir=log`（runs 为日志目录，必须指定），并按提示打开浏览器端口；
- 召唤画家：
```python
from torch.utils.tensorboard import SummaryWriter
summaryWriter = SummaryWriter(log_dir='log') # 实例化画家，log_dir 为日志目录
```

日志文件的组织方式：每运行一次（一个 "run"，即每实例化一个画家 SummaryWriter）都会产生一个新的日志文件。日志文件中记录了时间、设备等元信息与该画家的作画内容信息。**画布会呈现所有日志文件所画内容的并集**（可以在界面左下角选择部分的 "run" 显示），因此画家之间唯一的区别方式就是日志目录。


# TensorBoard 能画什么


官方文档：<https://pytorch.org/docs/stable/TensorBoard.html>

TensorBoard 通过 SummaryWriter 类的 `add_xx` 方法来画不同的内容，呈现在画布的各个**版块**上（上方选项卡），每个版块都有包含若干**子版块**（右方）；画布是交互式的而非静态，可以在画布上进一步调整可视化的效果，甚至导出（左方）。

![](TensorBoard.gif)

`add_xx` 方法有共同的参数：

- `tag` 参数：这部分内容的名字（字符串），必须指定。
- `walltime` 参数：默认为系统时间 `time.time()`。可以在 TensorBoard 界面红可视化这一信息。在画布 TIME SERIES 版块可以查看所有作画历史记录，会将调用的 `add_xx` 作出的内容按照该时间顺序排列。

以下是 TensorBoard 能画的东西（详细用法见文档，我只总结核心的东西）：

- 画曲线 `add_scalar`
  - 呈现在画布 SCALARS 版块；
  - 在曲线 `tag` 上添加一个坐标为 (global_step, scalar_value) 的点（注意 global_step 必须为整数）；
  - 不同的曲线画在不同的图上，一个子版块对应一个图；`add_scalars` 可以把多个曲线画在同一个图上；
  - 同理可以画直方图：`add_histogram`、二维图 `add_mesh` 等。
- 画图像：`add_image`
  - 呈现在画布 IMAGES 版块；
  - 在 IMAGES 板块的子版块 `tag` 上打印格式为 Tensor 类型的图像 img_tensor；
  - `add_images` 可以在一个子版块打印多个图像；
  - 同理可以画其他数据：音频 `add_audio`，文本 `add_text`，视频 `add_video`，Matplotlib 的 figure `add_figure`；
- 画表格：`add_hparams`
  - 呈现在画布 HPARAMS 版块；
  - 一次调用就添加一条表格记录（表格的一行）；
  - 传入字典，字典的键对应属性，值对应属性值；
  - 应传入两个字典，一个是自变量，一个是因变量；区分它们的意义是画布有对因变量做数据分析的交互功能；
- 画计算图：`add_graph`
  - 画 tensor 中存储的计算图；
  - 传入 `torch.nn.Module` 模型即可；
- 画 PR 曲线：`add_pr_curve`
  - 传入预测标签和真实标签，会自动计算准确率和召回率；
  - 一次调用就在 PR 坐标上添加一个点。
- 在低维空间（不超过 3 维）上展示高维数据：`add_embedding`
  - 采用的降维方法是 [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)，是数据可视化常用的降维方法；
  - 传入数据矩阵 Tensor；
  - 可以传入类别标签，则会以不同颜色显示类别；也可以传入其他类型的标签如字符串乃至图像，则图中的数据点会显示字符串或图像。


# TensorBoard 在深度学习中的用处


从上面看，TensorBoard 和通用的可视化工具的功能与逻辑差别不大。但它一开始就是为深度学习可视化设计的，主要兼容深度学习框架 Tensor 类型的数据，设计的可画内容都是深度学习需要可视化的。深度学习需要可视化：

- 数据：`add_images`, `add_video` 等；
- 学习曲线、loss 曲线等指标（随时间变化）：`add_scalar`；
- 网络结构图：用 `add_graph`；
- 不同超参数的效果比较：`add_hparams`（顾名思义，`add_hparams` 画表格就专门为超参数这事的）；
- 在低维空间可视化模型中间层 Embedding：`add_embedding`；
- PR 曲线：`add_pr_curve`。


深度学习的程序往往耗时很长，需要有存档机制，在代码中设置一些检查点（checkpoint），保存、加载训练到一半的模型参数等在[这篇笔记](https://pengxiang-wang.github.io/posts/readingnotes_Dive-into-DL_Part4/)中有详细的讨论。TensorBoard 也需要有存档机制，根据日志文件的组织方式——每次运行都会保存一个日志文件，画布会加载日志目录下的所有日志文件，所以我们无需手动保存、加载 TensorBoard 画图的进度。



# 七、超参数优化
（待更新）



# 学习曲线


过拟合、欠拟合的判断，