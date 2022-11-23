---
title: 配置 Python 环境备忘
date: 2021-04-19
categories: [科研]
tags: [日常管理, 技术]
img_path: /assets/img/
---



众所周知，Python 环境配置是一个大坑，很多人就莫名其妙地在电脑里装了大大小小十几个 Python 环境也不知道。看看这张梗图就明白了。

![python-meme](Python_meme.png)


我也经历过这样的事情。本文就简要记录一下自己的配置历史，并留一份当前的环境状态作为备忘。


# 一、个人电脑（Mac 系统）

## 吐槽

刚买回手里的这台 MacBook 后就迫不及待去官网把 Python 装了，用 pip 装了几个包一跑程序没问题，感觉美滋滋，就不再管了。

后来跑深度学习代码开始用上了 Conda，那段时间是真切地感受到坑了。安装完 Miniconda 后，在 VSCode 里发现了好几个 Python 解释器，便参考网上的教程一点点地卸载，没有记错的话，当时只保留系统了自带的 Python 2.7 以及 Miniconda。

过了段时间突然一看，发现系统自带的 Python 2.7 找不到了，吓我一跳，查了下发现是 Mac 系统更新到 Monterey 12.3 版本将自带的 Python 2 删除了。那么是更新成 Python 3 了，还是直接就删除了呢？网上各有各的说法。反正我的电脑上出现了一个让我疑惑的 Python 3.8.9，在 `/usr/bin/python3/`，它特别神奇，在系统里都找不到 Python.framework 框架，它还能正常运行，而且卸载也卸载不掉。看了[这篇博客](https://medium.com/@kailichou.edu/updated-remove-usr-bin-python3-or-not-69c63e8e62c0)，我吓尿了，还是留着吧，放在那里不用就好。


在我自己的电脑上，我还对一些扩展功能有刚需，如 Jupyter Notebook，IPython 等。它们本质上是 Python 的包，不需要单独安装，只要 pip intsall 就可以用了。

## Python 环境备忘

- 一个疑似系统自带的 Python 3.8.9，位于 `/usr/bin/`，勿使用，当祖宗供着；
- Miniconda3，位于 `~/Miniconda3/`，供平时使用；
- 环境变量中 python3 包括 Miniconda 中的 Python 3 与那个 Python 3.8.9（前者优先），python 是 Miniconda 中的 Python 2。无论敲 python3 还是 python，进入的都是 Miniconda 里的 Python 环境；
- 环境变量中的 pip, pip3 也是一样。无论敲 pip3 还是 pip，进入的都是 Miniconda 里的 Pip 环境。

## Conda 环境备忘

Mac 的定位是只做学习机或者跑一些简单的程序，不跑大型项目（例如 Mac 没有 Nvidia 显卡，装不了 CUDA，无法跑大型深度学习项目）。

- base：当作基本环境，只作临时使用，只在此环境中安装必要的通用的包，如 Jupyter Notebook，IPython，在创建新环境时都统一复制一份此环境；
- dl_study：学习、测试深度学习代码用，在 base 的基础上安装深度学习的包；
- spyder：爬虫程序的环境，在 base 的基础上安装爬虫相关包，如 requests, BeautifulSoup；

# 二、个人电脑（Windows 系统）

我的 Windows 游戏本是备用机，装有 GTX 960 显卡，可以跑深度学习项目。但我不把此当作主力机器，它只是偶尔测试一下代码用，主要还是用 Mac 远程连接服务器。

## Conda 环境备忘

本电脑只装一个 Miniconda 即可。同样地，base 环境中安装必要的通用的包，如 Jupyter Notebook，IPython，在创建新环境时都统一复制一份此环境。**base 以外的其他环境针对项目作临时用，不长期使用。**

# 三、服务器（Linux 系统）

服务器只跑大型项目，和 Windows 游戏本一样。

Conda 需要自行安装，无法共用其他账户的。一般用 Linux 命令行的命令来安装。

## Conda 环境备忘

只装一个 Miniconda 即可。同样地，base 环境中安装必要的通用的包，如 IPython，在创建新环境时都统一复制一份此环境（Jupyter Notebook 没必要装）。**base 以外的其他环境针对项目。**

目前我有两台服务器可用（组内、学院），都把 base 环境配置好，再根据项目需要安装项目环境。