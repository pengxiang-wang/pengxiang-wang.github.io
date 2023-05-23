---
title: PyTorch 学习笔记（五）：计算性能
date: 2022-02-11
categories: [科研]
tags: [读书笔记, 《动手学深度学习》, 机器学习, 技术]
img_path: /assets/img/
math: true
---

本文介绍 PyTorch 与计算性能有关的代码知识，包括如何使用 GPU、并行计算、多服务器计算等等。本文参考 [Dive into Deep Learning (PyTorch 版)](https://d2l.ai) 中的以下内容：
- 5.6 节：GPU；
- 第 12 章：计算性能；

------------------------------


# 深度学习与 GPU

众所周知，深度学习计算可以使用 GPU，往往能极大提高效率。GPU 用于深度学习时与其他任务不同，它更偏向于关注显存而不是算力（显存至少 8G 以上，见“效率问题”一节），因此也没有一个像桌面级显卡那样的天梯图可供参考。这里列举比较有名的型号：
- Nvidia GeForce 系列：个人电脑显卡，目前到 30 系列，价格万元以下，一般装在个人电脑上，跑一些深度学习程序够用；
- Nvidia RTX A6000：一般装在服务器上，价格比较昂贵，一块几万元；
- Nvidia Tesla V100：一般装在服务器上，价格比 A6000 贵；
- Nvidia Tesla A100：比 V100 更强大，目前成为大公司主流使用的。


跑深度学习的设备可以：
- 使用自己的电脑：
  - 不带 GPU 的电脑（如 MacBook）：没有 GPU 的加持，只用 CPU 跑会跑得很慢；
  - 带 GPU 的电脑（如 Windows 游戏本有 Nvidia 的独立显卡的）可以跑得快；
- 使用单台服务器（如组内的服务器）；
- 使用多台服务器（如大公司或机构公用的计算集群），方法见本文最后一节。





# PyTorch 与 CUDA

深度学习框架为我们提供了使用 GPU 硬件的高级 API，只需简单的代码即可使用 GPU 作深度学习的计算，甚至无需了解原理。

这些深度学习框架的高级 API 要与 GPU 打交道，但并不是直接打交道的。与普通的程序一样，通常调用操作系统提供的 SDK，操作系统与底层的 CPU 等硬件直接打交道；GPU 制造商也提供了类似 SDK 的接口，使用 GPU 的程序只需调用这个接口即可。Nvidia 公司的 GPU 提供的接口叫 **CUDA**，PyTorch 使用 GPU 的程序也是调用 CUDA 写的。所以要注意：
1. 要使用 PyTorch GPU 计算，首先显卡本身要支持 CUDA。这要求至少是 Nvidia 的显卡；
2. 要安装好 CUDA 才能使用 PyTorch。CUDA 理论上需要去 Nvidia 官网下载，但是不用担心，一般的机器只要有 Nvidia 的显卡，都是要安装显卡驱动的（否则显卡也没什么用），这个驱动里一般都带着 CUDA。没有 CUDA 的就安装驱动即可，也不用去手动下载。如果要手动安装 CUDA，要查好自己的 GPU 是否支持 CUDA，以及对应的 CUDA 版本。可以见 Nvidia 官方公布的对应列表：<https://developer.nvidia.com/cuda-gpus>；
3. CUDA 有很多版本，PyTorch 也为不同 CUDA 版本写了不同的代码，在安装的时候必须指定正确（和机器上的 CUDA 版本一致），否则在实际运行时由于不兼容，会跑不通。安装了 CUDA 的机器，可以直接用命令 `nvidia-smi` 查看 CUDA 版本（见下）;
4. PyTorch 版本也要和 CUDA 版本匹配（但不需要很严格），详见：<https://pytorch.org/get-started/previous-versions/>。

> 另外提一下，CUDA 是 GPU 完成**通用计算**任务的接口。GPU 一开始只是用来处理图像的，当时的接口只能完成图像处理任务；后来才开发出来其他用处，包括深度学习在内的并行计算，CUDA 就是也能完成这些任务的接口。
{: .prompt-info }

可以通过 `torch.cuda.is_available()` 验证是否配置成功。


# 查询与表示计算设备

安装了 CUDA 的，用终端命令 `nvidia-smi`（已配置环境变量）可以查看此机器的 GPU 信息，包括名称、型号、显存以及正在占用的程序等。每个计算设备（包括 GPU、CPU）用 `torch.device` 对象表示，此类构造接受一个字符串参数，字符串只允许以下几个：

- `torch.device('cpu')`：表示 CPU；
- `torch.device('cuda:i')`：表示第 i 个 GPU（i 是自然数）；
- `torch.device('cuda')`：等价于 `torch.device('cuda:0').

> CPU 是每台机器都有的，`torch.device('cpu')` 对象创建后总是与之绑定的；GPU 不是每台都有，而且可能也不会有多个，但仍然可以创建 `torch.device{'cuda:i'}` 对象，只是以后调用它时会报错指示 GPU 不存在。
> 书中写了两个方便的函数，可以自动查询 GPU，提供了简单的纠错机制，防止出现上述错误。纠错机制是通过一个查询可用 GPU 数量的 API：`torch.cuda.device_count()`。
> ```python
> def try_gpu(i=0):
>   '''如果可用，返回第 i 个 GPU 的设备对象（i 默认为 0）；否则返回 CPU 的设备对象'''
>   if torch.cuda.device_count() >= i+1:
>       return torch.device(f'cuda:{i}')
>   return torch.device('cpu')
> def try_all_gpus():
>   '''返回所有可用 GPU 的设备对象列表；否则返回 CPU 的设备对象（也是列表） '''
>   devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
>   return devices if devices else [torch.device('cpu')]
> ```
{: .prompt-warning }


# 将 Tensor 放到 GPU 上

在没有涉及 GPU 时，PyTorch 所有 Tensor 都是存放在内存里，此内存与 CPU 挂钩，做计算时将其运送到 CPU 中处理。所以使用 GPU 做计算要做的事情很简单：只需**将要用 GPU 计算的 Tensor 放到它的内存（称为显存）里**，计算时自动会运送到 GPU 中处理。

PyTorch 提供了简单的 API 将 Tensor 从不同计算设备来回转移：
- 在创建 Tensor 时只需为创建函数指定参数 `device=`，即可在指定设备创建，例：`X = torch.ones(2,3,device=torch.device('cuda')`；
- 

> 只有同一个计算设备的 Tensor 才能一起计算，在 PyTorch 中如果试图对来自不同设备的 Tensor 做计算，会直接报错。
{: .prompt-info }

深度学习计算涉及的 Tensor 有：**数据、模型参数**。只要所有的数据和参数都在同一计算设备上，就可以做训练了。
- 转移数据：
- 转移模型参数：模型和参数是绑定的，无需把参数从 `nn.Module` 对象取出来，PyTorch 的 API 形式上只需转移模型：`net.to(device=torch.device('cuda'))`


# 效率问题

以下是一些关于效率的 tips：

1. 在计算设备间转移 Tensor 时间开销是很大的，甚至比真正做计算都大很多。这也是 PyTorch 对不同设备计算直接报错，而不是尝试隐式转移到同一设备，这是为了防止用户发现不到问题造成大量的时间损失。
2. 选择哪个计算设备要根据实际情况而定。有的机器 GPU 性能还不如 CPU，那就不要盲目地用 GPU 计算了。
3. 而且在一个深度学习流程内，不是所有计算都放在 GPU 上效率才高。GPU 主要的优势是并行计算以及图像处理，主要是用在训练过程中，前面的预处理可能发挥不到 GPU 的优势。通常在临训练前才将数据转移到 GPU 上，但也有例外，如预处理需要处理图像。转移时机应根据实际情况把握。
4. GPU 在训练过程中优势主要发挥在一个 batch 的矩阵并行运算效率很高，所以 batch_size 开得越大越容易发挥计算上的优势。
5. 数据 batch 和模型参数都占用了显存。因此 batch_size 越大占用显存越大，容易出现显存不够用的情况。所以减少显存占用的一种方法是将 batch_size 调小，但会牺牲 GPU 的计算效率；另一种方法是减少模型参数、选用更小的模型。



# PyTorch 并行训练

上述涉及的都是在一个 GPU 上跑项目。当跑大型项目时，更需要在多个 GPU 上跑项目，甚至在多台机器上的多个 GPU，此时需要用到 PyTorch 并行训练的 API，前者称为 DP（Data Parallel），后者称为 DDP（Distributed Data Parallel）。

目前我的计算资源不够，暂时用不到，不打算学习总结这一部分。感兴趣的可以自行了解，见 PyTorch Tutorial：<https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>。



## 附：集群中的 GPU

使用多台服务器跑深度学习的情况，一般是大公司或机构公用的计算集群，即许多台服务器放在一个机房，它们通过统一的软件调度系统管理和使用。如果要使用，我们必须了解一些基本概念。

在集群中：
- 一个节点（node）表示一台机器；
- 一个节点即一台机器一般有多个 GPU；
- 一个分区（partition）表示一群节点，通常是按照功能等分类。例如我们数学学院的集群分 cpu 和 gpu 两区，cpu 区的节点机器 CPU 性能比较好（为了计算数学专业的同学跑 MATLAB 程序用），gpu 区 GPU 性能比较好（为我们做深度学习、图形学的同学）；公司的集群有的会根据 gpu 的性能来分。

集群的调度系统一般会限制用户不能直接登录具体的计算节点，而是先登录一个管理节点。在这个管理节点下，需要向调度系统申请使用计算节点。一般来说，用户需要提交一个申请命令，填写使用几个节点、GPU、多长时间等（一般不能指定使用哪个节点或哪个 GPU，这个是由调度系统分配的，不能我行我素），申请成功后即可以登录到计算节点。调度系统有一个非常厉害的功能就是，申请的计算节点和管理节点实时共用存储，内容是完全一样的，用户的数据只能存放在管理节点，并无需复制到计算节点，这一点要注意。其他关于调度系统的细节请自行查看你使用的集群的文档。


