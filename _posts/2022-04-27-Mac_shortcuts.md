---
title: MacOS 快捷键整理
date: 2022-04-27
categories: [其他]
tags: [技术]
img_path: /assets/img/
---

## 引言

快捷键是一个好东西，解决了很多用电脑时的痛点。但是去百度、B站、知乎等搜一下，就有大量带有营销包装性质的文章或视频。个人认为看这些东西也就图一乐，更多的人是在看节目/满足自己的好奇心/获得产品使用的幸福感，很难做到记住并实际使用。这些营销号也是利用了信息不对称，其实所有快捷键都在[苹果官方支持](https://support.apple.com/zh-cn/HT201236)中列出来了。这有点类似 CS:GO 的投掷物，后者也是在网上有大量的教程，但那条弹幕总是少不了：“进我的收藏夹吃灰吧！”

要让快捷键这个东西发挥作用，还是要长期观察自己的使用习惯，了解自己的需求，多动脑筋，尽量少而精。另外，有的时候快捷键反而不方便，比如比较懒的时候，一手托着腮，另一手用触控板，只想点点点，何必上快捷键找麻烦呢？


另有很多人会推荐第三方 App，美誉其为“神器”。但收费、兼容性问题、配置麻烦、跑路风险都是其缺点。这些第三方App包括但不限于 Alfred、Widgets。很多特色功能其实只要稍加思考就能找到原生系统里不错的快捷接口。举例来说，Alfred 就是一个高级版 Spotlight，它最亮眼的功能：打开应用、搜索引擎快捷搜索都可以在下文找到原生版的解决方案。

以下是整理的部分快捷键，它们完全解决了我用 Mac 快一年的痛点，且我认为已经完备，没有必要再掌握其他的快捷键。还是要重申，每个人情况不同，仅供参考，需要开发出自己的舒适区。





## 一、打开与关闭 App

打开与关闭 App 是最常用的操作，基本上每个人都有自己常用的 App，有些 App 还有**即时性**的特点：打开看一下即关闭。做法是：

- 打开：使用 Spotlight (`Cmd + Space`) 输入 App 的关键词，敲回车
- 关闭：快捷键 `Cmd + W`，等效于左上角红色关闭按钮。强制退出用 `Cmd + Q`

因此只需记住常用App的关键词即可，都是英文的前几个字母而已。总结如下：

| App  |关键词 | 常见使用场景 |
| :-: | :-: | :-: |
| 访达 （Finder.app） |   f   | 类似于打开我的电脑（`Win + E`） |
| Safari浏览器（Safari.app) |   sa    | 使用搜索引擎（下面第二节的基础） | 
| 微信 （WeChat.app） |    w, we, wec    | 打开，回消息，关闭 |
| 系统偏好设置（System Preferences.app) | sys | 打开某项常用设置，如我常用的 iPad 副屏 |
| 终端（Terminal.app) | ter | 程序员常用 |
| VS Code （VSCode.app） |    vsc   | 程序员常用 |



## 二、快速搜索

Spotlight 可以完全解决搜索问题，什么东西都可以搜，但这也是其缺点：不够精确，搜出来的东西需要挑选，浪费时间。它的这种设计个人认为更适合带有寻找灵感性质的搜索，但很少有人有这种习惯。最好是细分一下搜索的领域：
- 快速搜索本机文件（呼出 Finder 内的搜索）：`Option + Cmd + Space` （用空格的左边两个）
- 快速搜索引擎：用一的方法打开 Safari， `Cmd + T` （新标签页）
    - 第一种方法：`Cmd + L`（激活地址栏），输入网站前几个字母，敲回车（有时需要Tab一下激活搜索框）
    - 第二种方法：将常用的搜索引擎保持在书签栏前几位（例如我的是 Google, Baidu, Bing），使用快捷键 `Cmd + 1/2/3` 打开网页
    
## 三、Finder

Finder 是 Mac 的文件资源管理器，下面一些快捷键非常有用。请注意，用快捷键时一定要唤醒 Finder 哦！

- 后退 / 前进： `Cmd + [ / ]`  
- 上一级文件夹：`Cmd + 上`， 强烈建议使用，因为 Finder 不像 Windows 有向上一级的按钮
- 查看并复制路径：按住 `Option`，下面会出现路径，右键可复制；文件右键菜单也可以复制路径。此功能程序员常用
- 新开 Finder 窗口： `Cmd + N`，适用于已经打开了一个文件夹，想不关掉它再开一个的场景



## 四、Safari 浏览器

- 新标签页：`Cmd + T`
- 激活地址栏：`Cmd + L`，配合 `Cmd + T` 太完美了
- 关闭标签页：`Cmd + W` ，没错，Safari 这里只是关闭标签页而不是窗口
- 后退 / 前进： `Cmd + [ / ]`  
- 左右切换打开的标签页： `Cmd + Option + 左 / 右` （Option 用右边的）。其实还有一种方式是 `Ctrl （+ Shift） +  Tab` 类似于切换 App，但还是推荐这一种，如果 Option 用右边的 话，左手只有大拇指按在 Cmd 上，方便基于 Cmd 的快捷键操作例如，左右切换标签页的目的经常是在一堆打开的标签页挑些没用的用 `Cmd + W` 关掉
- 重新打开刚关闭的标签页：`Cmd + Shift + T`
- 开关书签边栏：`Ctrl + Command + 1`（Cmd 最好用右边的）。有时候开了边栏很占地方，想快速隐藏掉，也可用这个快捷键




## 五、文本编辑

Mac 系统缺失了很多 Windows 中一个键就可以解决的，但都可以通过组合键实现：
- 向前删除：`Fn + Delete`，相当于 Windows 的 `Delete`
- 光标放到此行开始 / 末尾处：`Option + 左 / 右`，相当于 Windows 的 `Home`，`End`

我的切换输入法方案是用 `caps lock` 切换中英文输入法，用 `Fn` 切换其他输入法（个人偶尔用到繁体中文、日语、俄语键盘）。`caps lock` 键不再用作大小写开关，因为大量输入大写字母的情形不多，少量输入使用 `Shift` 即可。

特殊符号的输入：
- 重音符号：英文输入法下，按住带重音的字母
- 中文常用字符：中文输入法下 `Shift + Option + B`
- Emoji表情：`Ctrl + Cmd + Space` 或 `Fn + E`
- 另外，按住 Option 或 Shift + Option 也可输入特殊的字符，但是需要记住一张字符映射表。特殊字符本身并不常用，没有必要，不推荐。



## 六、截图与录屏

Mac 不需要任何第三方截屏工具，自带的功能强大，完全够用。


- 截取整个屏幕并保存：`Cmd + Shift + 3`
- 截取选取的部分并保存：`Cmd + Shift + 4`
- 高级截取（内含录屏）：`Cmd + Shift + 5`


## 常识性的通用快捷键

这些列举的是其他常识性的，在 Windows 系统也常用的快捷键：
- 撤销：`Cmd + Z`
- 恢复：`Cmd + Shift + Z` （与 Windows 不同）
- 全选：`Cmd + A`
- 拷贝 + 粘贴：`Cmd + C`, `Cmd + V`
- 剪切 + 粘贴 （文件）：`Cmd + C`, `Option + Cmd + V` （与 Windows 不同）
- 剪切 + 粘贴 （文本）：`Cmd + X`, `Cmd + V` 
- 查找：`Cmd + F` （注意：替换没有固定的快捷键）
- 打开：`Cmd + O`
- 保存：`Cmd + S`



## 键鼠解决方案

在工位时我一般用机械键盘和鼠标，在这种键盘正下方不是触控板的情况下，有些快捷键就不太方便了。以下列举一些此场景的替代方案。

### 触控板手势

将触控板换成鼠标后，方便了一些操作，例如拖动；也平替了一些操作，例如滚动。但有些手势就不友好了，只能以其他方式替代：
- 左右切换桌面：四指左右滑动手势，改用 `Ctrl + 左/右`（`Ctrl` 用右边）。这样可以左手按快捷键，右手操作鼠标。
- 显示桌面：设置为右下角触发角，鼠标移到右下角即显示桌面。