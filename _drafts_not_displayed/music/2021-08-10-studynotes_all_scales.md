---
title: 音阶的分类方法
date: 2021-08-10
categories: [音乐]
tag: [音乐理论]
img_path: /assets/img/
math: true
---



本文介绍音阶，主要是推导如何系统地分类各种音阶。

构成一个音阶有几个要素：

- **主音**（tonic）：即音阶的起点音，决定了音阶的**调**（key）；
- **相邻音的排列方式**：音阶是从主音开始，按照一定的间隔排列方式，向上推演得到的。

主音决定绝对音高，排列方式决定色彩、和声功能（相对音高）。以下不讨论主音（可以都假设主音为C），而讨论有哪些排列方式。

一般要保证周期性，周期通常是一个八度，以下均作此假设（违反此假设的只有极其现代的自由爵士等类型的音乐）。按照十二平均律，一个八度之间有 12 个最小间隔（半音），所以音阶相当于把 12 拆成一些小间隔的和，根据拆解的方式不同得到各种音阶。根据排列组合知识，可以看成放隔板问题——在 12 棵树之间插入 $$n-1$$ 块隔板有多少组合方式，$$n$$ 音音阶就有多少个。计算公式为：

- 五音音阶有
- 六音音阶有：
- 七音音阶：
- 八音音阶：
- 九音音阶：
- 十二音音阶有 1 个：



有的比较常见，有名有姓，也有背后适用的音乐风格与故事，而有的则不常见。有名有姓的音阶请参考：<https://en.wikipedia.org/wiki/List_of_musical_scales_and_modes>


# 五音以上的音阶


## 对排列方式的限制

受该视频启发：<https://www.youtube.com/watch?v=Vq2xt2D3e3E>。我们对排列方式加如下几条限制：

- 间隔只包含 1（半音，H），2（全音，W），3（全音加半音，WH），3 以上跨度太大不考虑；
- 一些均匀性限制（为了使音阶好听）：
  - 两个 H 不相邻；
  - W, WH 不相邻；
  - WH, WH 不相邻。

注意第一个音和最后一个音也算相邻，是循环意义上的相邻。

这样得到的音阶都是七音或六音的。按照一套方法（调式）划分为几个族，以下每节介绍按照族来介绍：七音音阶共 4 个族，还有 2 族六音音阶和 1 族八音音阶。本章最后面证明列举了符合上述条件的所有情况。


### 自然大调族

自然大调的排列方式是 W-W-H-W-W-W-H，由此可以轮换衍生出 7 个不同的音阶，称为**调式**（mode），指通过轮换得到的一族音阶。调式的英文也很直观，就是数论中的模。这个排列方式循环周期是 7，自然大调音阶作为代表元。

以下每一族音阶不仅列出惯用名称与相邻音的排列方式，也列出 C 调音阶、 A 调音阶（前者全列，后者只对偏小调的列）与黑键少的容易的调的音阶。

|调式 |  惯用名  |  特色风格 | 相邻音的排列方式   |  C 调 |  A 调  | 容易的调与黑键个数 |
|:-:| ::- | :-: | :-:|:-:| -::|
|1 |**自然大调**，Ionian 调式 |  大调流行音乐 | W-W-H-W-W-W-H   | 1-2-3-4-5-6-7-1 |   | C 调（0个） |
|2|**Dorian 调式** |    | W-H-W-W-W-H-W |  1-2-b3-4-5-6-b7-1  |  6-7-1-2-3-**#4**-5-6  |  D 调（0个）：2-3-4-5-6-7-1-2 |
|3 |**Phrygian 调式**|    | H-W-W-W-H-W-W  |  1-b2-b3-4-5-b6-b7-1   |   |E 调（0个）：3-4-5-6-7-1-2-3 |
|4 |**Lydian 调式**|    | W-W-W-H-W-W-H |  1-2-3-**#4**-5-6-7-1|      |F 调（0个）：4-5-6-7-1-2-3-4   |
|5 |**Mixolydian 调式** |       | W-W-H-W-W-H-W  |  1-2-3-4-5-6-**b7**-1  |      |G 调（0个）：5-6-7-1-2-3-4-5    |
|6 |**自然小调**, Aeolian 调式| 小调流行音乐  |  W-H-W-W-H-W-W|  1-2-b3-4-5-b6-b7-1  | 6-7-1-2-3-4-5-6  | A 调（0个）    |
|7|**Locrian 调式**|     | H-W-W-H-W-W-W |   1-b2-b3-4-b5-b6-b7-1      |  |   B 调（0个）：7-1-2-3-4-5-6-7    |

> 可以发现，C 调 Ionian、D 调 Dorian、E 调 Phrygian …等音阶虽然音是一样的（白键），但主音不同，因此色彩不一样。因此对于音阶来说，主音也是很重要的。
{: .prompt-tip }

### 旋律小调族

旋律大调的排列方式循环周期是 7，衍生出以下 7 个音阶。

旋律小调与自然大调只差一个 3 音，因此衍生出的 7 个音阶与上表对应位置的音阶也只差某个音。

|调式|  惯用名  |  特色风格 | 相邻音的排列方式   |  C 调 |  A 调  | 容易的调与黑键个数 | 备注 | 
|:-:| ::- | :-: | :-:|:-:| :-: | :-: | -::|
|1|**旋律小调**| 斯拉夫国家如俄罗斯  |  W-H-W-W-W-W-H |  1-2-**b3**-4-5-6-7-1 |  6-7-1-2-3-**#4**-**#5**-6 | A 调（2个），C 调（1个）  | 下行一般要还原为自然小调：6-5-4-3-2-1-7-6 |
|2|  卡帕多细亚音阶  | 叙利亚音乐  |H-W-W-W-W-H-W|  1-b2-b3-4-5-6-b7-1   |    6-b7-1-2-3-#4-5-6   | D 调（1个）：2-b3-4-5-6-7-1-2 | 即 Dorian b2 音阶（Dorian 调式 2 音降半音，下同）、Phrygian #6 音阶 |
|3|      |    | W-W-W-W-H-W-H|  1-2-3-#4-#5-6-7-1    |   | F 调（1个）：4-5-6-7-#1-2-3-4  |  即增 Lydian 音阶（Lydian 调式 5 音升半音为增五度） |
|4| 原声音阶，泛音阶，蓬迪科尼西音阶 |  图瓦（俄罗斯）音乐  |W-W-W-H-W-H-W|   1-2-3-#4-5-6-b7-1    |   |    F 调（1个）：4-5-6-7-1-2-b3-4 | 即 Lydian b7 音阶，也称 Lydian Dominant  |
|5| **旋律大调**, 印度音阶，奥林匹亚音阶 |  印度传统音乐 |  W-W-H-W-H-W-W|  1-2-3-4-5-b6-b7-1 |  6-7-**#1**-2-3-4-5-6  |  A 调（1个）   | 也称大 Aeolian 音阶（Aeolian 调式 3 音升半音为大三度）、Mixolydian b6 音阶|
|6|  半减（Half-Diminished）音阶, 西西弗斯音阶 |     |W-H-W-H-W-W-W | 1-2-b3-4-b5-b6-b7-1|   6-7-1-2-**b3**-4-5-6   |  A 调（1个）  |  即 Locrian ♮2 音阶；叫半减音阶原因是主音七和弦为半减七和弦| 
|7|  变形（**Altered**）音阶，帕拉米蒂音阶  |  | H-W-H-W-W-W-W| 1-b2-b3-3(b4)-b5-b6-b7-1  |    |  B 调（1个）：7-1-2-**b3**-4-5-6-7  | 也称超级 Locrian 音阶，因为 Locrian 把自然大调降了5个音，而它把 6 个音全降了  |


### 和声小调族

和声小调的排列方式循环周期是 7，衍生出以下 7 个音阶。

|调式|  惯用名  |  特色风格 | 相邻音的排列方式   |  C 调 |  A 调  | 容易的调与黑键个数 | 备注 | 
|:-:| ::- | :-: | :-:|:-:| :-: | :-: | -::|
|1|**和声小调** |   | W-H-W-W-H-WH-H | 1-2-b3-4-5-b6-7-1  | 6-7-1-2-3-4-**#5**-6 | A 调（1个） | 即 Aeolian #7 音阶 |
|2|     |    |H-W-W-H-WH-H-W | 1-b2-b3-4-b5-6-b7-1 |   |B 调（1个）：7-1-2-3-4-#5-6-7   | 即 Locrian ♮6 音阶 |
|3|   |     |W-W-H-WH-H-W-H | 1-2-3-4-**#5**-6-7-1  |    | C 调（1个）  |  即 Ionian #5 音阶，也称自然增大调 |
|4| 乌克兰小调，罗马尼亚小调  |   东欧音乐   |W-H-WH-H-W-H-W| 1-2-b3-#4-5-6-7-1  |  6-7-#1-2-3-#4-5-6  |  D 调（1个）：2-3-4-#5-6-7-1-2 | 即 Dorian #4 音阶 |
|5|  **阿拉伯音阶** |  阿拉伯音乐   | H-WH-H-W-H-W-W| 1-b2-3-4-5-b6-b7-1 |  | E 调（1个）：3-4-#5-6-7-1-2-3  | 也称**大 Phrygian 音阶**、Phrygian Dominant  |
|6|     |   |WH-H-W-H-W-W-H| 1-#2-3-#4-5-6-7-1  |   | F 调（1个）：4-#5-6-7-1-2-3-4  | 即 Lydian #2 音阶 |
|7|     |        |H-W-H-W-W-H-WH  | 1-b2-b3-3(b4)-b5-b6-6(bb7)-1 |     |    |  即 Altered bb7 音阶 |


### 和声大调族

和声大调的排列方式循环周期是 7，衍生出以下 7 个音阶。

|调式|  惯用名  |  特色风格 | 相邻音的排列方式   |  C 调 |  A 调  | 容易的调与黑键个数 | 备注 |
|:-:| ::- | :-: | :-:|:-:| :-: | :-: | -::|
|1| **和声大调** | 印度卡纳提克音乐 | W-W-H-W-H-WH-H| 1-2-3-4-5-**b6**-7-1 |  |  C 调（1个）  |  即 Ionian b6 音阶  |
|2|    |  | W-H-W-H-WH-H-W| 1-2-b3-4-b5-6-b7-1 | 6-7-1-2-b3-#4-5-6   |  D 调（1个）：2-3-4-5-b6-7-1-2  | 即 Dorian b5 音阶 |
|3|  |  | H-W-H-WH-H-W-W|  1-b2-b3-3(b4)-5-b6-b7-1   |    | E 调（1个）：3-4-5-b6-7-1-2-3  |  即 Phrygian b4 音阶|
|4|    |   |W-H-WH-H-W-W-H| 1-2-b3-#4-5-6-7-1  |     | F 调（1个）：4-5-b6-7-1-2-3-4| 即 Lydian b3 音阶 |
|5|    |    | H-WH-H-W-W-H-W| 1-b2-3-4-5-6-b7-1  |   | G 调（1个）：5-b6-7-1-2-3-4-5| 即 Mixolydian b2 音阶 | 
|6|    |   | WH-H-W-W-H-W-H|  1-#2-3-#4-#5-6-7-1 |   |   | 即增 Lydian #2 音阶 | 
|7|   |   | H-W-W-H-W-H-WH|  1-b2-b3-4-b5-b6-6(bb7)-1   |    |  B 调（1个）：7-1-2-3-4-5-b6-7  | 即 Locrian bb7 音阶|


> 这里讨论一下大调与小调。**大调**与**小调**就是为一些特殊的音阶起的名字，狭义上指自然大调、自然小调，广义上通常称开头（下四音列）是 W-W-H 的为大调、W-H-W 的为小调。
> 
> **关系大小调**是指大调音阶与小调音阶在同一个调式循环里的（即上面的同一族里的）。可以看到自然大小调、旋律大小调是关系的，而和声大小调不是关系的。另外，下面五声音阶、Blues 音阶也有关系的大小调。
{: .prompt-tip }

### 非七音音阶

以下音阶不是七音音阶，而是六音、八音等音阶。

第一个是全音音阶，是六音音阶，循环周期为 1：
|  惯用名  |  特色风格 | 相邻音的排列方式   |  C 调   | 容易的调与黑键个数 |
| ::- | :-: | :-:|:-:| :-: | 
|全音音阶（六音音阶）|  梦幻、仙境   |   W-W-W-W-W-W   | 1-2-3-#4-#5-#6-1  | 都一样 |  


第二个是增音阶，是六音音阶，循环周期为 2：
| 相邻音的排列方式   |  C 调 | 容易的调与黑键个数 | 
| :-:|:-:| :-: | :-: | -::| 都一样 |
| WH-H-WH-H-WH-H |    1-#2-3-5-#5-7-1   |   都一样 |
| H-WH-H-WH-H-WH |   1-#1-3-4-#5-6-1 | 都一样 |

第三个是减音阶，是八音音阶，循环周期为 2：
 | 相邻音的排列方式   |  C 调 |  容易的调与黑键个数 |
| :-:|:-:| :-: |
|W-H-W-H-W-H-W-H | 1-2-b3-4-b5-b6-6-7-1 | 都一样 |
|H-W-H-W-H-W-H-W |1-b2-b3-3-b4-5-6-b7-1 |都一样 |

### 分析完备性

我们对数字 12 的拆解方式作分类。

先看只有 W, H 出现的。H 必须是偶数个：

- $$12  = 2\times 6$$：即全音音阶；
- $$12 = 2\times 5+ 1\times 2$$：H 最近间隔 2 个、1 个，即自然大调族、旋律小调族；
- $$12 = 2\times 4 + 1 \times 4$$：W 和 H 只能插空排列，即减音阶；

再看包含 WH 的。WH 和 H 的个数之和为偶数：

- $$12 = 2\times 4 + 1\times 1 + 3\times 1$$：WH 和 H 相邻，另外一边只能和 WH 相邻，不满足条件；
- $$12 = 2\times 3 + 1\times 3 + 3\times 1$$：一共 3 个 H，不能相邻，中间要用 W 和 唯一的 1个 WH 隔开。根据先用 W 还是 WH 隔开，得到和声小调、和声大调族；
- $$12 = 2\times 2 + 1\times 2 + 3\times 2$$：WH 太多了，H 太少，不可能满足 WH 与 H 不相邻的条件；
- $$12 = 1\times 3 + 3 \times 3 $$：WH 和 H 只能间隔排列，即增音阶。

## 放宽限制

如果放宽如上的限制，可以得到更多其他的音阶，但大部分不是很实用。例如一些特殊的七音音阶。

双和声大调族，放宽了两个 H 不相邻的限制。
|调式|  惯用名  |  特色风格 | 相邻音的排列方式 |  C 调   | 容易的调与黑键个数 | 备注 |
|:-:| ::- | :-: | :-:|:-:| :-: |  -::| 
|1| 双和声大调 / 拜占庭音阶 / 吉卜赛大调| 拜占庭音乐，弗拉明戈   | H-WH-H-W-H-WH-H   | 1-b2-3-4-5-b6-7-1| C 调（2个），E 调（2个）：   | 可看成阿拉伯音阶（大 Phrygian 音阶）与和声大调音阶的结合 | 
|2|    |     | WH-H-W-H-WH-H-H |  1-#2-3-#4-5-#6-7-1 |  D调（2个）：    |    即 Lydian #2 #6 音阶  |
|3|    |     |  H-W-H-WH-H-H-WH |    1-b2-b3-3(b4)-5-b6-6(bb7)-8     |  E 调（2 个）：3-4-5-b6-7-1-b2-3  |     也称终极 Phrygian 音阶    | 
|4|双和声小调  /匈牙利小调 / 吉卜赛小调 / 阿尔及利亚音阶  |     | W-H-WH-H-H-WH-H  | 1-2-b3-#4-5-b6-7-1 |    |      |
|5| 东方音阶   |     | H-WH-H-H-WH-H-W |  1-b2-3-4-b5-6-b7-1 |      |      |
|6|    |     | WH-H-H-WH-H-W-H |  1-#2-3-4-#5-6-7-1 |      | 即 Ionian #2 #5 音阶    |
|7|    |     | H-H-WH-H-W-H-WH |  1-b2-2(bb3)-4-(b5)-(b6)-6(bb7)-1 |      | 即 Locrian bb3 bb7 音阶     |



|印度坡尔维音阶 |    |H-WH-W-H-H-WH-H | 两个 H 不相邻，W 与 WH 不相邻  |  1-b2-3-#4-5-b6-7-1 |    | 
|印度陀地音阶 |   | H-W-WH-H-H-WH-H |  两个 H 不相邻，W 与 WH 不相邻     | 1-b2-b3-#4-5-b6-7-1 |  ｜      |


神秘音阶：H-WH-W-W-W-H-H，1 – ♭2 – 3 – ♯4 – ♯5 – ♯6 – 7
吉普赛 W-H-WH-H-H-W-W

那不勒斯小调族，放宽了 W 与 W、H 与 H 不相邻的限制。
|调式|  惯用名  |  特色风格 | 相邻音的排列方式 |  C 调   | 容易的调与黑键个数 | 备注 |
|:-:| ::- | :-: | :-:|:-:| :-: |  -::| 
|1 |那不勒斯小调|  |  H-W-W-W-H-WH-H |	1-b2-b3-4-5-b6-7-1
|2 |Lydian ♯6|	1-2-3-#4-5-#6-7-1	
|3| Mixolydian Augmented|	1-2-3-4-#5-6-b7-1|
|4|	Romani Minor
(or Aeolian/Natural Minor ♯4)| 1-2-b3-#4-5-b6-b7-1|
|5|	Locrian Dominant|	1	♭2	3	4	♭5	♭6	♭7	8	
|6|Ionian/Major ♯2|	1	♯2	3	4	5	6	7	8	
|7|Ultralocrian/Altered Diminished bb3|	1	♭2	double flat3	♭4	♭5	♭6	double flat7	8	



那不勒斯大调族，放宽了 W 与 W、H 与 H 不相邻的限制。
|调式|  惯用名  |  特色风格 | 相邻音的排列方式 |  C 调   | 容易的调与黑键个数 | 备注 |
|:-:| ::- | :-: | :-:|:-:| :-: |  -::| 
|1|	那不勒斯大调|	  |  H-W-W-W-W-W-H  | 1-b2-b3-4-5-6-7-1	
|2|Leading Whole Tone (or Lydian Augmented ♯6)| 1	2	3	♯4	♯5	♯6	7	8	
|3|	Lydian Augmented Dominant|	1	2	3	♯4	♯5	6	♭7	8	
|4|	Lydian Dominant ♭6|	1	2	3	♯4	5	♭6	♭7	8	
|5|	| 阿拉伯音乐|  W-W-H-H-W-W-W | 	1-2-3-4-b5-b6-b7-1	  |  | 也称大 Locrian 音阶 |
|6|	Half-Diminished ♭4 (or Altered Dominant ♯2)| 1	2	♭3	♭4	♭5	♭6	♭7	8	|
|7|	Altered Dominant bb3|	1	♭2	double flat3	♭4	♭5	♭6	♭7	8	|

波斯音阶族，放宽了 H 与 H、W 与 WH 不相邻的限制。
|调式|  惯用名  |  特色风格 | 相邻音的排列方式 |  C 调   | 容易的调与黑键个数 | 备注 |
|:-:| ::- | :-: | :-:|:-:| :-: |  -::| 
|1	| 波斯音阶 | H-WH-H-H-W-WH-H|	1-b2-3-4-b5-b6-7-1| 
|2| Ionian #2 #6|	1-#2-3-4-5-#6-7-1|
|3|	Ultraphrygian bb3|	1-b2-bb3-b4-5-b6-bb7-1|
|4|	Todi Thaat	|1-b2-b3-#4-5-b6-7-1 |
|5|	Lydian #3 #6 |	1-2-#3-#4-5-#6-7-1 | 
|6|	Mixolydian Augmented ♯2|	1-#2-3-4-#5-6-b7-1|
|7|	Chromatic Hypophrygian Inverse|	1-b2-bb3-4-b5-bb6-bb7-1|



以及大部分的六音音阶，包括 Blues 音阶、三全音音阶、普罗米修斯音阶等，详见[维基百科：六音音阶](https://en.wikipedia.org/wiki/Hexatonic_scale)、

甚至[八音音阶](https://en.wikipedia.org/wiki/Octatonic_scale)、八音以上的音阶乃至十二音的半音音阶。


## 七音音阶的四音列分解

（前 3 个间隔称为下四音列，后 3 个间隔称为上四音列，二者连接起来就是一个七音音阶）。四音列的类型。
Chromatic	1 1 1	H-H-H
Diminished	1 2 1	H-W-H
Gypsy [sic]	2 1 3	W-H-3H
Harmonic	1 3 1	H-3H-H
Major	2 2 1	W-W-H
Minor	2 1 2	W-H-W
Phrygian/
Upper minor	1 2 2	H-W-W
Whole tone	2 2 2	W-W-W

# 五音音阶

如果继续放宽上面限制的第一条，即允许跨度更大的音程——4、5 等，可以使得音阶中的音更少，例如本章要介绍的五音音阶。当然，可以像上面那种分类方式一样作完备的分类，但是组合方式是在是太多了，而且组合出来的情况多数都不常用，所以本章只挑几个常用的讲解。

值得一提的是，这些常用的一般可以看成上面常用的**七音音阶去掉两个音**，使得它的属性和对应的七音音阶很像。


以下把间隔 3 记作 m（小三度），4 记作 M（大三度）。

## 经典五声调式

自然大调音阶去掉 4音 与 7 音，得到大调五声音阶，循环周期为 5，衍生出以下 5 个音阶。这种调式是中国传统调式，中国古代叫宫、商、角、徴、羽；日本受中国影响，称这类调式为雅乐调式，包括吕调式、律调式。

|  惯用名  |   排列方式   |   C 调 |  用的黑键最少的调及个数 |
| :-: | :-: | :-:|
|**大调五声音阶**，宫调，(日本）吕（ryo）调式（吕旋法） | W-W-m-W-m | 1-2-3-5-6-1   | C 调, 0 |
|商调  |  W-m-W-m-W |   2-3-5-6-1-2    |  D 调, 0 | 
| 角调| m-W-m-W-W|  3-5-6-1-2-3  |  E 调, 0 |
| 徴调，(日本）律（ritsu）调式（律旋法）| W-m-W-W-m| 5-6-1-2-3-5 |   G 调, 0    |
| **小调五声音阶**，羽调| m-W-W-m-W | 6-1-2-3-5-6  | A 调, 0  |

### Blues 音阶

将五声调式中两个相邻的 W 中的第二个拆解成两个 H，相当于在该位置中间加了一个音（这个音就是 Blues 音乐的特色音），这样五音音阶变为六音音阶，就是 **Blues 音阶**。

按说上面五种都可以得到对应的 Blues 音阶，但是常用的只有这两种。

|  惯用名  |   排列方式   |   C 调 |  A 调 ｜   用的黑键最少的调及个数 |
| :-: | :-: | :-:|
|小调 Blues 音阶 | m-W-H-H-m-W | 1-2-3-5-6-1   | C 调, 0 |
|大调 Blues 音阶  |  W-m-W-m-W |   2-3-5-6-1-2    |  D 调, 0 | 

- ：，特色音位于 #4 (b5) 位置；
- ：W-H-H-m-W-m，特色音位于 #2 (b3) 位置。

Blues 音阶虽然是六声音阶，但不符合上一章的规则（W 与 WH 相邻了），而且它是由五声衍生的，所以放在这里讲。

## 日本调式

自然大调音阶拿掉 2 音与 5 音，得到日本的俗乐调式，循环周期为 5，衍生出以下 5 个音阶。

|  惯用名  |   排列方式   |   用的黑键最少的调及个数 |
| :-: | :-: | :-:|
|  平（hira）调子 |   W-H-M-H-M   |  A 调, 0   |    
| 岩户(iwato）调子 |   H-M-H-M-W    |  B 调, 0  | 
|  阳旋法，田舍节（inaka bushi）调子 |  M-H-M-W-H     |  C 调, 0|
|  本云井（hon kumoi）调子  |  H-M-W-H-M |  E 调, 0 | 
| 阴旋法，都节（miyako bushi）调子  |   M-W-H-M-H    |  F 调, 0 |  
   
 > 以上出现的“旋法”本身的意思是日本筝乐器的调音方式。
 {: .prompt-tip }




## 东南亚调式


自然大调音阶拿掉 2 音与 6 音，也得到一个循环周期为 5 的调式，衍生出以下 5 个音阶。这些音阶很有东南亚风味。

|  惯用名  |   排列方式   |   用的黑键最少的调及个数 |
| :-: | :-: | :-:|
| (日本）琉球旋法  |  M-H-W-M-H  | C 调, 0 |  
|   |  H-W-M-H-M    | E 调, 0  |    
|   |  W-M-H-M-H     | F 调, 0  |    
|   |  M-H-M-H-W|  G 调, 0 |  
|   |   H-M-H-W-M    |  B 调, 0  |  


# 其它怪异的音阶



- 四声。三全音：12 = 3\times 4

free jazz。


## Mixolydian 混合蓝调音阶

先不管这个名字是怎么来的，直接上音阶：

A 调 Mixolydian 混合蓝调音阶： 6  7 1 #1 2 b3 3  #4 5 6


为什么要加这三个音？看我下面的思路历程：
- 混合了大调 A 和弦 体现在 1#
- 剩下的看上去是又把 4 音和 7音加回来了。那为什么加的是 4#，而不是4？A 大调。
- 那最后的5音为什么不变成#5？因为是向蓝调音阶妥协了，蓝调是基础。所以看上去混的不是 A 大调音阶，而是A Mixolydian 音阶（此音阶名字的由来）。


总结起来就是，这个音阶本质上是在 A 小调的蓝调音阶中，混入 A 大调的音符，使其出现大调的感觉。


所以，对待这个音阶态度不能像以前一样，不是完全弹出这些音阶中的音就可以的。蓝调音阶仍是基础，只是混进去的其他音给乐曲增加大调色彩。


另一个观点是把 A 小调蓝调变成 A 大调蓝调。










根据爵士理论，音阶与和弦挂钩，和弦建立在音阶几级几级上的，其他任何音乐都不例外。



