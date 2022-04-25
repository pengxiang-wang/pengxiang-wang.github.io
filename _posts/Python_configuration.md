---
title: 配置 Python 环境备忘
author: Shawn Wang
date: 2022-04-11 13:00:00 +0800
categories: [其他]
tags: [技术]
math: true
---



众所周知，Python 环境配置是一个大坑，很多人就莫名其妙地在电脑里装了大大小小十几个 Python 环境也不知道。看看这张梗图就明白了。

![python-meme](/assets/img/2022-03-02_1.png)


我也经历过这样的事情。本文就简要记录一下自己的配置历史，并留一份当前的环境状态作为备忘。


一、个人电脑（Mac系统）

刚买回手里的这台 MacBook 后就迫不及待去官网把 Python 装了，用 pip 装了几个包一跑程序没问题，感觉美滋滋，就不再管了。

后来跑深度学习代码开始用上了 Conda，那段时间是真切地感受到坑了。安装完 Miniconda 后，在 VSCode 里发现了好几个 Python 解释器，便参考网上的教程一点点地卸载，没有记错的话，当时只保留系统了自带的 Python 2.7 以及 Miniconda。

过了段时间突然一看，发现系统自带的 Python 2.7 找不到了，吓我一跳，查了下发现是 Mac 系统更新到 Monterey 12.3 版本将自带的 Python 2 删除了。那么是更新成 Python 3 了，还是直接就删除了呢？网上各有各的说法。反正我的电脑上出现了一个让我疑惑的 Python 3.8.9，在 `/usr/bin/python3/`，它特别神奇，在系统里都找不到 Python.framework 框架，它还能正常运行，而且卸载也卸载不掉。看了[这篇博客](https://medium.com/@kailichou.edu/updated-remove-usr-bin-python3-or-not-69c63e8e62c0)，我吓尿了，还是留着吧，放在那里不用就好。


当前此电脑的环境备忘：

- 一个疑似系统自带的 Python 3.8.9，位于 `/usr/bin/`，勿使用，当祖宗供着
- Miniconda3，位于 `~/Miniconda3/`，使用
- 环境变量中 python3 包括 Miniconda 中的 Python 3 与那个 Python 3.8.9（前者优先），python 是 Miniconda 中的 Python 2。无论敲 python3 还是 python，进入的都是 Miniconda 里的 Python 环境
- 环境变量中的 pip, pip3 也是一样。无论敲 pip3 还是 pip，进入的都是 Miniconda 里的 Pip 环境



二、服务器（Linux系统）
