---
title: 爬虫项目通用架构
date: 2022-08-14
categories: [有趣的事情]
tags: [技术]
img_path: /assets/img/
---


多年之前读本科的时候买了[崔庆才](https://cuiqingcai.com)的《Python3 网络爬虫开发实战》，会了一点点爬虫的技术，当时就是随性写写写几十行的小代码，能随便爬点维基百科之类的。这些东西学的很零散，简直是拾起来就忘。

最近又有爬虫需要了，这里有意识地关注方法论，搜了一下“爬虫架构”，不出意外，发现爬虫项目都是遵循统一的流程（或称架构），例如[菜鸟教程里讲的](https://www.runoob.com/w3cnote/python-spider-intro.html)。本文想系统地整理一下爬虫的固定流程，每个阶段负责做什么，有什么工具可供使用，以便以后爬虫时能立刻理顺逻辑；另外也提供一个通用的爬虫项目模版，以便以后可以抄抄作业。

# 爬虫的固定流程

网络爬虫遵循的固定流程可以如下叙述：

1. 访问待爬网站，下载网页的 HTML 代码；
2. 将需要的内容从 HTML 文本中解析出来；
3. 将解析内容处理好后，输出或写入文件；
4. 根据规则（如网站 URL 的命名规律）得到下一个待爬网站，重复步骤 1,2,3，直到没有待爬的网站。

按照面向对象封装的思想，各部分可以由以下几个功能模块负责：

- **URL 管理器**：负责管理所有待爬的 URL，通常存放于集合中（因为不想重复爬一个网站），生成下一个待爬网站 URL 的规则在逻辑上包含在这里；
- **HTML 下载器**：将待爬 URL 的 HTML 下载下来；
- **HTML 解析器**：将需要的内容从 HTML 文本中解析出来；
- **数据管理器**：将解析内容处理好后，存储到变量、输出或写入文件；
- **调度器**：总指挥，规定了以上功能模块如何调度（按照什么顺序执行，数据从哪里到哪里）以实现完整的爬虫（依据上面所述的流程），有点像主函数的作用。

# 通用爬虫模版

该模版参考此文：<https://cloud.tencent.com/developer/article/1423611>。我将此文给出的模版作了一些简化和抽象。

```python
from urllib import response
import requests
import sys
from config import *

class URLManager:
    '''
    URL 管理器
    '''
    def __init__(self):
        self.urls = set() # 存储已经爬过的 URL，用集合实现

    def root2first(self, root_url):
        '''
        从首页找到第一个待爬的 URL，找不到则返回空
        '''
        if notfound:
            return None

        return first_url

    def next(self, url):
        '''
        寻找下一个待爬的 URL，找不到则返回空
        '''
        if notfound:
            return None

        self.urls.add(next_url)
        return next_url


class HTMLDownload:
    '''
    HTML 下载器
    '''
    def download(self, url):
        '''
        从 URL 中下载 HTML 文本
        '''
        if url is None:
            print('HTML下载器：URL为空，HTML未下载')
            return None
        response = requests.get(url)
        if response.status_code != '200':
            print(f'HTML下载器：URL<{url}>不存在，HTML未下载')
            return None
        response.encoding = 'utf-8'
        html = response.text
        return html

class HTMLParser:
    '''
    HTML 解析器
    '''
    def parse(self, html, url):
        '''
        从 HTML 文本中解析出想要的内容
        '''
        
        return data


class DataManager:
    '''
    数据管理器
    '''
    def __init__(self):
        self.dataset = [] # 存储爬到的数据

    def cleaning(self, data):
        '''
        清洗爬到的数据
        '''
        
        return data

    def store(self, data):
        '''
        存储爬到的数据
        '''
        self.dataset.append(data)

    def output(self, data):
        '''
        输出爬到的数据
        ''' 

    def process(self, data):

        data = self.cleaning(data)
        self.store(data) # （可选）
        self.output(data) # （可选）


class Spider:
    '''
    调度器
    '''
    def __init__(self):
        '''
        调度器包含以上四个功能模块：URL 管理器、HTML 下载器、解析器、数据管理器
        '''
        self.manager = URLManager()
        self.downloader = HTMLDownload()
        self.parser = HTMLParser()
        self.datamanager = DataManager()

    def crawl(self, root_url):
        '''
        调度完成爬虫
        '''
        url = self.manager.root2first(root_url)
        while url:
            html = self.downloader.download(url)
            data = self.parser.parse(html, url)
            self.datamanager.process(data)
            

if __name__ == '__main__':
    spider = Spider()
    spider.crawl(ROOT_URL)
```


