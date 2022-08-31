---
title: 读书笔记：《动手学深度学习》Part 4：复杂神经网络入门，GPU 的使用
date: 2022-02-11
categories: [科研]
tags: [读书笔记, 《动手学深度学习》, 技术]
img_path: /assets/img/
math: true
---


## 书籍信息 

### [Dive into Deep Learning (PyTorch 版)](https://d2l.ai)
- 作者：亚马逊团队
- 配套课程：[李沐](https://space.bilibili.com/1567748478/) 主讲，视频上传于 B 站。链接：<https://c.d2l.ai/zh-v2/>
- 本 Part 内容：第 5 章全部内容，包括自定义模型、参数管理、读写文件、GPU 的使用。本部分内容是 CNN、RNN 等大型神经网络的基础，训练大型网络脱离不了本部分学习的概念与机制。

------------------------------

# 自定义模型

目前为止，我们见到的所有高级 API 定义的模型全是使用 PyTorch 现有的模版：由 `nn.Sequential()` 包裹的 `nn.Linear()`, `nn.Flatten()`，用它们定义出的模型非常默认，不够灵活。在科研中，我们有时候就是要设计不同的网络结构、训练方式，这种默认的就无法满足需求了，需要自定义模型。学会本节可以掌握编写更加复杂的网络。

在逻辑层面上，所有网络模型都是由**块**（block）组成的，块与块之间可以有各种顺序、嵌套、并列等关系。块中包含一个或多个**层**（layer）。在 PyTorch 的语义中，模型最小单位不是神经元而是层。

在代码写法上，所有块、层都是 `nn.Module` 对象，包括 PyTorch 现有的 `nn.Linear()`,`nn.Flatten()` 甚至 `nn.Sequential()`。下面是通用的自定义 `nn.Module` 对象的写法。

官方文档：<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>

## 通用 nn.Module 模版

```python
class MyModule(nn.Module):
    def __init__(self, ???)
        super().__init__()
        self.参数 = ...???
    
    def forward(self, X):
        return ... # X 的表达式
```

它的逻辑是，只要定义好前向传播 `forward` 函数，里面包含的是 torch 运算，再确保把这些运算的参数（即网络参数）封装到此模型的类中即可。`forward` 函数是核心，在自定义 `nn.Module` 对象时必须要写，`__init__()` 函数只是一个将 `forward` 函数所需变量绑定于此类的容器。下面做一个小试验，体会其重要性：
```python
M = nn.Module()
M()
M.forward()
```
这段代码第二、三行都会报 `NotImplementedError`，提示 `forward` 函数未定义。为什么会这样？`nn.Module` 模块的源代码解释清楚了：
```python
def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

class Module:
    #...
    forward: Callable[..., Any] = _forward_unimplemented

    def _call_impl(self, *input, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        #...
        result = forward_call(*input, **kwargs)
        #...
        return result

    __call__ : Callable[..., Any] = _call_impl
    #...
```
`nn.Module` 设计的机制就是要求继承时必须重写一个 `forward` 函数。此外，可以直接调用 `nn.Module` 对象，其实就是给定输入$$x$$前向传播一遍得到预测结果，它由 `__call__()` 方法定义（见 [Python 笔记]()：类特殊方法），而实现中可以看到出现了 `self.forward`，所以不写 `forward` 函数，在训练时前向传播也会报 `forward` 函数未定义的错误。

另外，在构造函数中需要调用父类 `nn.Module` 的函数：`super().__init__()`，为了把 `nn.Module` 定义的一些实例属性继承过来，只能这样写。写 `nn.Module` 对象时都要调用一下，否则会因缺少里面的属性报变量未定义的错误。感兴趣可以看看这些实例属性是什么：
```python
class Module:
    # ...
    def __init__(self) -> None:
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._non_persistent_buffers_set: Set[str] = set()
        self._backward_hooks: Dict[int, Callable] = OrderedDict()
        self._is_full_backward_hook = None
        self._forward_hooks: Dict[int, Callable] = OrderedDict()
        self._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._state_dict_hooks: Dict[int, Callable] = OrderedDict()
        self._load_state_dict_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._load_state_dict_post_hooks: Dict[int, Callable] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
    #...
```



有了以上模版，我们可以使用 `nn.Module` 写一个层，也可以写一个块，甚至块组成的一整个模型，非常灵活，取决于 `__init__()` 和 `forward` 函数怎么写。以下每种情况分别给出两段代码：一段是调用现有的模版，另一段是自己继承 `nn.Module` 手写出来的；这两段代码写出来的效果是一样的。

## 自定义层
例 1：全连接层。
```python
net = nn.Linear(8, 128)
```
```python
class MyLinear(nn.Module):
    def __init__(self, input_num, output_num):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
    def forward():
        return torch.matmul(X, self.weight.data) + self.bias.data

net = MyLinear(8, 128)
```
例 2：ReLU 激活函数层，它是一个不带参数的层。
```python
relu = nn.ReLU()
```
```python
class ReLULayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return nn.functional.relu(X)
```

参数打包成 `nn.Parameter` 类后，直接定义为实例属性，`forward` 函数直接拿来用。实际使用时，一般很少自定义层，一般的网络都是使用那些常用层如全连接层、卷积层等，然后按照下面的方式组合成块。

## 自定义块
例：多层感知机（MLP）。
```python
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10))
```
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.out = nn.Linear(256, 10)
    def forward(self, X):
        return self.out(nn.functional.relu(self.hidden(X)))

net = MLP()
```
也就是说，一些定义了层的 `nn.Module` 对象能以这种方式嵌套进定义了块的对象。

上面两者还有微小的区别：前者使用 `nn.Sequential`，用下标 `net[0]`,`net[1]`索引各层，还能把激活函数当作层索引到；后者的层是放在实例属性上的，需要用 `.` 来索引。

`nn.Sequential` 是一种特殊的 `nn.Module` 对象，如上所述它能起到顺序连接各层、充当列表的效果。它的原理可以参考书中的简单复现，由此例可以体会到自定义 `nn.Module` 对象的灵活性：
```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
```
可以看到，它可以传入任意多个 `nn.Module` 对象（可变参数 `args`），将其顺序存储在 `_modules`（之前见过是 `nn.Module` 的实例属性，这里就派上用场了，是一个 `collections.OrderedDict` 容器），然后在 `forward` 函数中顺序复合到输入 X 上。（注意，这样使得激活函数也可作为可变参数传入。）

## 自定义块组成的模型

`nn.Module` 对象是可以一层一层地递归嵌套地定义的。以下是一个稍复杂的例子：
```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.ReLU(), nn.Linear(16, 10))
```

现在思考，为什么可以这样嵌套呢？因为在前向传播时，会一层一层递归地调用被嵌套模块的 `forward` 函数。例如此例，调用时 `chimera(X)` 时，首先会调用 `nn.Sequential` 的 `forward` 函数，即依次调用 `X = NestMLP()(X)`, `X = nn.ReLU()(X)`, `X = nn.Linear(16, 10)(X)`，前面说过每一层都会调用各自 `forward` 函数，例如先调用 `NestMLP()` 的 `forward`，其中调用了 `self.net()`, `self.linear()`，它们又会调用里面的 `nn.Sequential(...)`,`nn.Linear(32, 16)` 的 `forward` 函数。如此递归下去，直到遇到真正层里面的参数，例如 `self.linear` 里面的 `weight`。这种 `forward` 函数递归过程会把嵌套的每一块、层的参数都遍历到，从而能反向传播。此递归调用相当于遍历下面这颗树：

![6](NestMLP_called.png)

有人会问，为何不用简单的：
```python
chimera = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 16), nn.ReLU(),
                        nn.Linear(16, 10))
```
这就要从方便性的角度考虑了。这种方便不只在命名上（见下）。在逻辑上前者是将网络分成了几个块，例如本例有点像 NestMLP 是特征提取器，后面的全连接层是分类器的意思，假如以后想换个特征提取器更方便。


最后讨论一下命名问题。模型的每个块、层都有自己的名字（类似变量的命名空间），且可以通过以这个名字命名的实例属性访问。在这种封装的模型类中，嵌套关系的存在使得各块、层有树形关系。例如上面的 NestMLP 模型各块、层的名字如下：

![6](NestMLP_namespace.png)

> `print()` 函数能以一种规整的方式打印出网络结构（是由类特殊方法 `__print__()` 和 `__repr__()` 定义的，见 [Python 笔记]()），会显示各块、层的名字、网络结构。也可以使用 Tensorboard 等工具可视化，见 [Tensorboard 笔记]()。
{: .prompt-tip }


# 参数管理

这一节系统讲解了与 `nn.Module` 对象定义的模型参数有关的操作。上面已经看到，用 `nn.Module` 对象封装的模型参数都属于 `nn.Parameter` 类（[文档](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)）。此类在构造时接受两个参数：

- `data`：参数数据，Tensor 或 `nn.Parameter` 类型；
- `requires_grad`：指定 data 是否需要梯度。

不用管它与 Tensor 到底有什么区别，反正它是封装起来的，一定封装了一些有用的机制。要直接取得 `nn.Parameter` 类封装的 Tensor，应访问实例属性 `.data`。



## 访问参数

参数是最里层模型内部的东西，即树外挂的叶子结点，

![6](NestMLP_parameters_called.png)

直接按照图中绿色叶子结点的调用方式即可访问单层参数。

也可以模型的访问所有参数，与上面同理，其算法也是递归地遍历树的叶子。其 API 有：

- `.parameters()` 方法：返回一个生成器，print 无法直接显示，需要遍历其元素 print；另外有 `.named_parameters()` 方法，返回生成器生成的是 (参数名字, 参数数据) 对。
- `.state_dict()` 方法：返回一个 `collections.OrderedDict` 类型，字典键值为 {参数名字:参数数据}， print 可以显示。

## 参数初始化

这里讨论封装在 `nn.Parameter` 中的参数的初始化。当然可以直接取出 data 属性，对其赋值或修改。

更好用的是能直接对 `nn.Parameter` 对象操作的 API。PyTorch 提供了很多初始化函数，作用在 `nn.Parameter` 对象上。这些函数定义在 `nn.init` 模块中，以下列举几个常用的，其他的详见文档：<https://pytorch.org/docs/stable/nn.init.html>。

- `nn.init.normal(tensor, mean, std)`：从正态分布 $$N(mean, std)$$ 初始化
- `nn.init.constant(tensor, val)`：全部以常量 val 初始化
- `nn.init.uniform(tensor, a, b)`：从均匀分布 $$U(a,b)$$ 初始化
- `nn.init.xavier_uniform(tensor, gain)`, `nn.init.xavier_normal(tensor, gain)`：Xavier 初始化，见 Bengio 等人论文 [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
- `nn.init.kaiming_uniform(tensor, a, mode)`, `nn.init.kaiming_normal(tensor, a, mode)`：何恺明的初始化，见论文 [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf).

> `nn.init` 系列函数也可以直接作用在 Tensor 上。
{: .prompt-info }

`nn.init` 系列函数的作用方式通常是打包成一个 `init_ parameters` 函数，间接地通过 `apply` 函数作用到模型参数上。这样做的好处是方便维护代码。


# 读写文件

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


# 使用 GPU

众所周知，深度学习计算可以使用 GPU，往往能极大提高效率。深度学习框架为我们提供了使用 GPU 硬件的高级 API，只需简单的代码即可使用 GPU 作深度学习的计算，甚至无需了解原理。

> 这些深度学习框架的高级 API 要与 GPU 打交道，但并不是直接打交道的。与普通的程序一样，通常调用操作系统提供的 SDK，操作系统与底层的 CPU 等硬件直接打交道；GPU 制造商也提供了类似 SDK 的接口，使用 GPU 的程序只需调用这个接口即可。
> Nvidia 公司的 GPU 提供的接口叫 **CUDA**，PyTorch 使用 GPU 的程序也是调用 CUDA 写的。所以：
> 1. 要安装好 CUDA 才能使用 PyTorch。不用担心，在安装 PyTorch 时选择了 GPU 版本，CUDA 会自动安装，无需去 Nvidia 官网手动下载 CUDA；
> 2. 要 PyTorch 将深度学习任务用 GPU 计算，应当用支持 CUDA 的 GPU。而且 CUDA 有好几个版本，对应支持不同的 GPU。在安装 CUDA（即安装 PyTorch）前一定要查好自己的 GPU 是否支持 CUDA，以及对应的 CUDA 版本。见 Nvidia 官方公布的对应列表：<https://developer.nvidia.com/cuda-gpus>.
> 另外提一下，CUDA 是 GPU 完成**通用计算**任务的接口。GPU 一开始只是用来处理图像的，当时的接口只能完成图像处理任务；后来才开发出来其他用处，包括深度学习在内的并行计算，CUDA 就是也能完成这些任务的接口。
{: .prompt-info }

## 查询与表示计算设备

安装了 CUDA 后，终端命令 `nvidia-smi`（已配置环境变量）可以查看此机器的 GPU 信息，包括名称、型号、显存以及正在占用的程序等。每个计算设备（包括 GPU、CPU）用 `torch.device` 对象表示，此类构造接受一个字符串参数，字符串只允许以下几个：

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


## 将 Tensor 放到 GPU 上

在没有涉及 GPU 时，PyTorch 所有 Tensor 都是存放在内存里，此内存与 CPU 挂钩，做计算时将其运送到 CPU 中处理。所以使用 GPU 做计算要做的事情很简单：只需**将要用 GPU 计算的 Tensor 放到它的内存（称为显存）里**，计算时自动会运送到 GPU 中处理。

PyTorch 提供了简单的 API 将 Tensor 从不同计算设备来回转移：
- 在创建 Tensor 时只需为创建函数指定参数 `device=`，即可在指定设备创建，例：`X = torch.ones(2,3,device=torch.device('cuda')`；
- 

> 只有同一个计算设备的 Tensor 才能一起计算，在 PyTorch 中如果试图对来自不同设备的 Tensor 做计算，会直接报错。
{: .prompt-info }

深度学习计算涉及的 Tensor 有：**数据、模型参数**。只要所有的数据和参数都在同一计算设备上，就可以做训练了。
- 转移数据：
- 转移模型参数：模型和参数是绑定的，无需把参数从 `nn.Module` 对象取出来，PyTorch 的 API 形式上只需转移模型：`net.to(device=torch.device('cuda'))`


## 效率问题

以下是一些关于效率的 tips：

1. 在计算设备间转移 Tensor 时间开销是很大的，甚至比真正做计算都大很多。这也是 PyTorch 对不同设备计算直接报错，而不是尝试隐式转移到同一设备，这是为了防止用户发现不到问题造成大量的时间损失。
2. 选择哪个计算设备要根据实际情况而定。有的机器 GPU 性能还不如 CPU，那就不要盲目地用 GPU 计算了。
3. 而且在一个深度学习流程内，不是所有计算都放在 GPU 上效率才高。GPU 主要的优势是并行计算以及图像处理，主要是用在训练过程中，前面的预处理可能发挥不到 GPU 的优势。通常在临训练前才将数据转移到 GPU 上，但也有例外，如预处理需要处理图像。转移时机应根据实际情况把握。
4. GPU 在训练过程中优势主要发挥在一个 batch 的矩阵并行运算效率很高，所以 batch_size 开得越大越容易发挥计算上的优势。
5. 数据 batch 和模型参数都占用了显存。因此 batch_size 越大占用显存越大，容易出现显存不够用的情况。所以减少显存占用的一种方法是将 batch_size 调小，但会牺牲 GPU 的计算效率；另一种方法是减少模型参数、选用更小的模型。
