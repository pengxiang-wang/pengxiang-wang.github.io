---
title: 代码笔记：IIRC (Incremental Implicitly-Refined Classification)
author: Shawn Wang
date: 2022-04-09
categories: [科研]
tags: [代码笔记, 持续学习]
math: true
---


## 论文信息 



### [IIRC: Incremental Implicitly-Refined Classification](https://openaccess.thecvf.com/content/CVPR2021/papers/Abdelsalam_IIRC_Incremental_Implicitly-Refined_Classification_CVPR_2021_paper.pdf)


- 会议：CVPR 2021
- 作者：
    - Mohamed Abdelsalam, Mojtaba Faramarzi - 蒙特利尔大学，[Mila-Quebec AI Institute](https://mila.quebec)，博士生
    - [Shagun Sodhani](https://shagunsodhani.com) - Facebook
    - [Sarath Chandar](http://sarathchandar.in) - 蒙特利尔工程学院，加拿大高等研究院（CIFAR），[Mila-Quebec AI Institute](https://mila.quebec)
- 内容：提了一个持续学习的新场景，是类别增量学习的推广：允许数据包含多个由粗到细的标签，允许旧数据携带细标签进入新任务。

### [文章的代码在这里！](https://github.com/chandar-lab/IIRC) [文档在这里！](https://iirc.readthedocs.io)

------------------------------



## API概述

此论文代码共分为2个部分：iirc package 和 lifelong_methods package，它们都属于 Python 的第三方库。

IIRC 是一个持续学习的 Benchmark，包括了为新场景重新整理的数据集、重新制定的训练+测试流程以及重新定义的评价指标。这些API都是在 iirc package 中定义的：

- Dataset类（iirc.lifelong_dataset.torch_dataset.Dataset）：将普通的数据集封装成一整个持续学习场景的数据集，即在一个Dataset类中规定了是什么持续学习场景、有几个任务、每个任务涉及哪些类别、各类包含哪些数据等等。具体用法见[此处](https://iirc.readthedocs.io/en/latest/iirc_tutorial.html)

- 训练+测试流程：作者没有打包成一个高级的API，而是指导你应将代码写成如下形式：

```
example_model = lifelong_methods.methods.example.Model(args)  # replace example with whatever module is there

for task in tasks:
    task_data <- load here the task data
    
    # This method initializes anything that needs to be inizialized at the beginning of each task
    example_model.prepare_model_for_new_task(task_data, **kwargs) 

    for epoch in epochs:
        # Training
        for minibatch in task_data:
            # This is where the training happens
            predictions, loss = example_model.observe(minibatch)

        # This is where anything that needs to be done after each epoch should be done, if any
        example_model.consolidate_epoch_knowledge(**kwargs) 
    
    # This is where anything that needs to be done after the task is done takes place
    example_model.consolidate_task_knowledge(**kwargs)

    # Inference
    # This is where the inference happens
    predictions = example_model(inference_data_batch)
```

但是要求模型 `example_model` 是一个打包好的类，必须是 `lifelong_methods.methods.base_method.BaseMethod` 的继承，且要定义好其中用到的函数。具体用法见[此处](https://iirc.readthedocs.io/en/latest/lifelong_methods_guide.html)。



为了测试这个 Benchmark，作者也亲自写了一些持续学习的 method （即上面的 `example_model` ），写好了放在 lifelong_methods package 里方便调用。当然，自己的模型还是要自己写，而且要写成 `example_model` 的规范，才能用他们这个 Benchmark 测试。


