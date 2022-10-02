---
title: 自动检测网站无效链接工具
date: 2022-08-14
categories: [有趣的事情]
tags: [技术]
img_path: /assets/img/
---


最近有朋友反映我的网站上有一些指向网站上其他文章的超链接是无效的，一般是两种情况：

- 点进去是 404：一般是我后来把文章的 Markdown 文件改名了，导致文章的 URL 也改了名；当然也可能是打错了；
- 点不动：没错，是我故意空着的，先鸽一鸽以后再补票哈哈哈。

现在网站上的文章多了起来，每次挨个文章找这些空链接非常麻烦，于是想写个小工具帮我自动检测无效链接，并帮我填入正确的。

我的文章放在几个目录中，已发布的在 `_post` 目录；未发布的就在自己本地的其他目录。小工具实现两个功能其实很简单：

- 自动检测：就是对目录里的文件进行文本匹配，然后访问匹配到的链接试试看是否有效。由于 Markdown 语法把链接都用括号包围起来了，所以匹配用很简单的正则表达式就能完成。访问网站只需要 requests 库，看返回码是不是 404 即可；
- 自动纠正：从其他已有的文件名里找一个和此笔记最相符的。这里写固定的规则就可以了，用不着机器学习那一套。

```python



```