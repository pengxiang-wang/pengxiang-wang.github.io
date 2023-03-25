---
title: Python 命令行解析参数
date: 2023-03-20
categories: [科研]
tags: [学习笔记, 技术, Python]
img_path: /assets/img/
---

本文介绍 Python 程序的命令行参数的定义和使用方法，主要为参数解析库 `argparse` 、处理配置文件 YAML 的库 `PyYAML` 的接口。

当运行的代码有多个参数并需要多次运行时，简单的处理方法是将参数放在代码里面的全局变量中，每次运行前修改变量的值。但这样需要大量的手动操作，非常费力且不够优雅；且这些参数本身在逻辑上属于程序的输入，将其与代码主体分离是更加合理的。

argparse 官方文档：<https://docs.python.org/zh-cn/3/library/argparse.html>


# sys.argv

在 Python 中，最简单的实现方式是借助 `sys` 库下的 `argv` 变量，这个在 [Python 笔记]()中已经见过：

```shell
??/??/python ??/??/prog.py arg1 arg2 ...
```

在程序内这些参数通过 `sys.argv` 来接收（需要 import sys 模块）。它是一个字符串列表，存放了解释器的执行命令的所有参数。以上为例，`sys.argv` 的内容为 `['??/??/prog.py','arg1','arg2',...]`。

# argparse

上述方法有很多缺点，例如：
- 无法指定参数类型：只能接受字符串输入，其他类型需要通过与字符串的类型转换函数转换；
- 只能实现位置参数，处理默认参数、可变参数等类型的参数非常麻烦；
- 没有独立健全的报错机制，需要自行处理输入格式错误；
- ...

上面的功能 `argparse` 库都有相当完善的接口实现了。`argparse` 模板如下：

```python
import argparse

parser = argparse.ArgumentParser(description='prog.py help')
parser.add_argument('--arg1', ...)
parser.add_argument('--arg2', ...)

# 调用命令行参数
args = parser.parse_args()
args.arg1; args.arg2...
```
```shell
??/??/python ??/??/prog.py --arg1=?? --arg2=?? ...
```
每调用 `add_argument` 方法一次，就定义了一个命令行参数，`add_argument` 方法的第一个参数 `name` 是参数名。

在代码中调用的方式是用 `args = parse_args()` 方法获取命令行参数 `args`，这个 `args` 是一个 argparse 封装的容器类型（argparse.Namespace）（类似于 Python 字典），调用格式为 `args.参数名`。

`add_argument` 函数的参数定义了关于该命令行参数的细节。

## 指定数据类型

`add_argument` 的 `type` 参数指定了对应命令行参数的数据类型，取值就是 Python 的各种类型类（不是字符串）。默认为字符串。


## 参数列表

`argparse` 让 `.py` 文件的命令行参数拥有 shell 指令那样的参数格式（见 Linux 笔记）：
- 位置参数：是默认的参数类型，按照 add_argument 的添加顺序；
  - 注意这里的位置参数允许不传，此时参数为 None，如果强制要求该参数应当设置 `add_argument` 的 `required` 为 True；
  - 允许接受（固定的）多个参数，在 `add_argument` 的 `nargs` 参数中设置数量，以列表的形式存放。
- 默认参数：参数的默认值在 `add_argument` 的 `default` 参数中设置；
- 可变参数：在 `add_argument` 的 `nargs` 参数中设置
  - 设置为 `'*'` 或 `'+'`，它们二者的区别是是否允许 0 个参数。
- 命名关键字参数：必须以 `参数名=` 的形式传的参数，在参数名中前加 `--` 或 `-` 来标识，分别称为长名与短名；
  - 允许规定多个名（包括长名和短名），并列地列在 `add_argument` 开头的参数中即可，在 `args.参数名` 调用时，这个参数名取为第一个长名（如果没有长名，则取第一个短名）。

上面可能没有特别详细地解释解析命令行参数的规则，但已经够用了，完整的规则都定义在 `parse.args()` 方法中，请参考文档：<https://docs.python.org/zh-cn/3/library/argparse.html#the-parse-args-method>

一般来说，除了简单的程序使用位置参数，一般常用的是命名关键字参数。

### 子命令实现参数分组

上述定义的程序的逻辑是，程序只做了一件事，这件事（命令）有一系列参数。有时候程序需要**做多件事（命令）**，简单的方式可以，指定一个参数来选择，但是有时候命令要求的参数可能不共用甚至完全不相同。`argparse` 提供了利用**子命令对参数隔离与分组**的功能，通过“子 parser”来实现：
```python
import argparse

parser = argparse.ArgumentParser(description='prog.py help')
parser.add_argument('--arg1', ...)
...

subparsers = parser.add_subparsers(help='sub-command help')

parser_cmd1 = subparsers.add_parser('cmd1', help='cmd1 help')
parser_cmd1.add_argument('--cmd1_arg1', ...)
...

parser_cmd2 = subparsers.add_parser('cmd2', help='cmd2 help')
parser_cmd2.add_argument('--cmd2_arg1', ...)
...

# 调用命令行参数
args = parser.parse_args()
args.arg1

cmd1_args = parser_cmd1.parse_args()
cmd1_args.cmd1_arg1

...
```
这里为该程序创建了几个子命令：cmd1,cmd2,...，它们对应于自己的 parser，该 parser 下的参数只有在打出相应的子命令时才会生效，从而实现参数隔离与分组：
```shell
??/??/python ??/??/prog.py --arg1=?? cmd1 --cmd1_arg1=?? ...
```
最外层 parser 的是全局的参数，例如上面的 `--arg1`。

子 parser 也是 ArgumentParser 类，从而可以递归下去，形成更细分的子命令，从而形成一颗子命令树。





## 生成帮助文档

`argparse` 库可以自动生成如何使用 py 文件命令行参数的帮助文档，因为它自动创建了一个 `-h` 或 `--help` 命令行参数：
```shell
??/??/python（解释器） ??/??/prog.py -h（或--help）
```
调用此命令即可在命令行显示如下形式的文档：
```shell
usage: test_argparser.py [-h] [--arg1 ARG1] {cmd1,cmd2} ...

prog.py help

positional arguments:
  {cmd1,cmd2}  sub-command help
    cmd1       cmd1 help
    cmd2       cmd2 help

optional arguments:
  -h, --help   show this help message and exit
  --arg1 ARG1  arg1
```
调用子命令的 -h 道理是一样的：
```shell
??/??/python（解释器） ??/??/prog.py cmd1 -h（或--help）
```
```shell
usage: test_argparser.py cmd1 [-h] [--cmd1_arg1 CMD1_ARG1]

optional arguments:
  -h, --help            show this help message and exit
  --cmd1_arg1 CMD1_ARG1
                        cmd1_arg1_help
```

这些帮助信息的文本定义在上面接口的参数里：
- ArgumentParser 的 description 参数：整个帮助文档的描述；
- add_argument 的 help 参数：每个参数的使用方法；
- add_subparsers 的 help 参数：子命令的总体使用方法；
- add_parser 的 help 参数：每个子命令的使用方法。

帮助文档也可以通过 `argparse` 的 IO 接口输出为别的形式，如文本文件等，详见[文档](https://docs.python.org/zh-cn/3/library/argparse.html#printing-help)。


# argparse + YAML 配置文件

使用 `argparse` 已经相当优雅，但还有一个最大的缺点，就是在参数数量特别多时，在命令行中敲一长串。笨方法可以把长命令复制出来，在调用的时候粘贴又显得不够优雅，而且在一行内打出来所有的参数也不直观。解决方案是**把长串参数放在格式化的文本文件中，在调用时解析文本文件里面的参数**。

对于 Python，常用的这种文本文件格式是 [YAML]()，它的语法我不在这儿写了，放一个[菜鸟教程](https://www.runoob.com/w3cnote/yaml-intro.html)在这儿，只要知道 YAML 可以表示 Python 大部分的数据类型与结构，用的时候现查即可。YAML 对应的解析库是 `PyYAML` （注意不是 Python 自带的，需要安装）。这个库只是一个文本解析器，它不和命令行参数打交道，需要配合 argparse 传入 YAML 文件路径，将路径交给 PyYAML 解析参数。模板如下：

```python
import argparse, yaml

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', ...)

with open(parser.cfg) as f:
    config = yaml.load(f.read()) 

```
```shell
??/??/python ??/??/test.py --cfg=(YAML文件路径)
```
YAML 文件中的参数通过 load 方法传递到了程序的 config 变量中，它是一个字典，通过 `config['参数']` 的方式调用。

可以看到，YAML 可以与 argparse 定义的普通参数结合，例如在 argparse 的普通参数中定义重要的、通用的，而在 YAML 中定义次要的、麻烦的。YAML 也可以定义多个。


<br>

总结来看，需要根据自己的需求，选择以上三种方式。三种方式依次实现了更完善的功能，但配置和写代码的成本也依次增加。
- 如果只做临时用，只需要一两个简单的位置参数，选第一种；
- 参数类型复杂，要分类，需要完善地交给用户，选第二种；
- 参数类型复杂、个数极多，选第三种。