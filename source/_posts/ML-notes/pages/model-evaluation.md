---
title: ML-notes:模型评估与选择
date: 2024-04-15 08:36:52
tags: [ML-notes,ML,notes]
---

# 2 模型评估与选择
##  1. <a name='1'>经验误差与过拟合</a>
### 1.1 <a name='1-1'>经验 Experiences</a>
> Experience = The data we have for training the machine learning model.

对于特定机器学习任务，**已存在的可利用数据**即是解决该机器学习任务的**经验**。而在这个大数据时代，大数据=丰富经验=训练更好的机器学习模型。

### 1.2 <a name='1-2'>数据划分</a>
通常我们会对获取到的数据进行数据划分，也就是我们绪论提到的一些学术用语：

<div align=center>
<img src="/img/pics/2-1.png" />
</div>

- **训练集 Training Set**：用来训练模型或确定参数。
- **测试集 Testing Set**：测试已经训练好的模型的推广能力。
- **验证集 Validation Set**（可选）：是模型训练过程中单独留出的样本集，它可以用于调整模型的超参数和用于对模型的能力进行初步评估。用来做模型选择（Model Selection），即模型的最终优化及选择。

### 1.3 <a name='1-3'>误差与精度</a>
#### 误差 Error
我们将学习器对样本的**实际预测结果与样本的真实值之间的差异**称之为**误差（error）**。而误差包含三类：
- **训练误差 training error 或 经验误差 empirical error**：学习器在训练集上的误差。
- **泛化误差generalization error**：学习器在新样本的误差，也就是实际误差。
- **测试误差Testing Error**：学习器在测试集上的误差，用来近似泛化误差。因为对于实际误差总是变化着的，所以我们一般用测试误差来近似泛化误差。

#### 经验误差vs.泛化误差
- **经验误差**：在训练集上的误差，训练误差
- **泛化误差**：在新样本上的误差，实际误差

> 是不是两个误差都越小越好？

> 不，因为会出现过拟合 overfitting

### 1.4 <a name='1-4'>过拟合(overfitting) vs. 欠拟合(underfitting)</a>
<div align=center>
<img src="/img/pics/2-2.png" />
</div>

我们以叶子为例子，如图所示，过拟合的话，会认为树叶必须有锯齿，欠拟合的话，认为绿色都是树叶。所以误差过小或过大，都会与预期结果不符。

#### 我的定义
- **过拟合（over-fitting）**：所建立的机器学习模型或者是深度学习模型在训练样本中表现得过于优越，导致在验证集以及测试集中表现不佳。
- **欠拟合（under-fitting）**：训练样本被提取的特征较少，导致训练出来的模型不能很好地匹配，导致表现不佳，甚至样本本身都无法高效的识别。
> 非官方解释

<u>所以我们发现 一个模型的评估与选择是多么的重要。</u>
## 2. <a name='2'>模型选择</a>

模型选择主要是回答三个问题，也就是下面我们讲的三个问题：
- 如何获得测试结果？ -> **2.1 评估方法**
- 如何评估性能优劣？ -> **2.2 性能度量**
- 如何判断实质差别？ -> **2.3 比较检验**

### 2.0 <a name='2-0'>典型的机器学习过程</a>
我认为通常机器学习分五步：
- **得到数据**：机器学习过程中，最重要并且困难的通常都是**清理和预处理数据**。除开直接得到的数据外，一般数据都需要数据挖掘、数据清洗、特征工程、数据转换、特征提取等一系列数据准备工作。
- **模型选择**：得到数据后，我们需要针对不同的问题和任务选取恰当的模型，而**模型就是一组函数的集合**。
- **确定标准**：选取到适合的模型后，我们需要确定一个衡量标准，也就是我们通常说的 **损失函数（Loss Function）** 来衡量模型函数的好坏。
- **得到函数**：最后也是困难的也是这一步 **Pick the “Best” Function**，我们需要进行多次训练，从众多函数中得到 Loss 最小的一个。通常会使用一些 梯度下降法、最小二乘法等技巧来选择。
- **测试**：学习得到“最好”的函数后，需要在**新样本上进行测试**，只有在新样本上表现很好，才算是一个“好”的函数。

#### 调参
##### 概念
大多数学习算法都有些**参数(parameter) 需要设定**，参数配置不同，学得模型的性能往往有显著差别，这就是通常所说的**参数调节或调参** (parameter tuning)。
##### 如何调参

学习算法的很多参数是在实数范围内取值，因此，对每种参数取值都训练出模型来是不可行的。常用的做法是：**对每个参数选定一个范围和步长λ**，这样使得学习的过程变得可行。

例如：假定算法有 k 个参数，每个参数仅考虑 j 个候选值，这样对每一组训练/测试集就会有 k^j 个模型需考察。
> 当重新调参之后就要重新训练模型，所以调参的时间代价是很大的。
##### 验证集
我们在[数据划分](#1-2)的时候有说到**验证集**，有验证集的模型训练步骤是这样的：先在训练集上训练，验证集上做测试，重复以上步骤选出最好的模型，把训练集和验证集并到一起做训练，在测试集上最后测试。

也就是：算法参数设定后，要用“训练集+验证集”重新训练最终模型。而测试集，是用来评估最终模型的。

###  2.1 <a name='2-1'>评估方法</a>

评估方法描述了**如何获得测试结果**。而它的关键便是：怎么获得“测试集”(test set)。一般来说，我们应该保证**测试集应该与训练集“互斥”**。

所以我们常见的方法如下：
- **留出法(hold-out)**
- **交叉验证法(cross validation)**
- **自助法(bootstrap)**

####  2.1.1 <a name='2-1-1'>留出法</a>

<div align=center>
<img src="/img/pics/2-3.png" />
</div>

将数据集 D 划分为**两个互斥的集合**，一个作为训练集 S，一个作为测试集 T，满足 D=S∪T 且 S∩T=∅ ，常见的划分为：大约2/3-4/5的样本用作训练，剩下的用作测试。
> 留出就是说，数据集的一部分作为训练集，剩下的留给测试集

**注意事项：**
- **保持数据分布一致性**
>（避免因数据划分过程引入额外的偏差而对最终结果产生影响，例如分类任务中**保持两个数据集（S、T）样本的类别比例相似**，也就是**分层采样**)
- **多次重复划分**
>(单次使用留出法得到的估计结果往往不够稳定可靠，需要多次实验)
- **测试集不能太大、不能太小**
>(**测试集小**时，评估结果的方差就比较大，**训练集小**时，评估结果的偏差较大。常见值：1/5~1/3，测试集至少应含30个样例
[Mitchell, 1997])
####  2.1.2 <a name='2-1-2'>交叉验证法</a>

<div align=center>
<img src="/img/pics/2-4.png" />
</div>

将数据集D**等分为k份相互不重叠的子集**，每次**取1份子集作为测试集，其余子集作为训练集**。重复k次，直至所有子集都作为测试集进行过一次实验评估，最后取平均值。我们也称之为 **k 折交叉认证**。

**同样为保证数据分布一致性**，即为减少因样本划分不同而引入的差别，一般会随机使用不同划分重复p次，比如10次10折交叉验证。
####  2.1.3 <a name='2-1-3'>自助法</a>

自助法比较好理解，**又称可放回采样**。和我们学的概率论的可放回抽取类似：

假设有m个样本，每次采一个作为测试集，采到的概率是1/m，采不到的概率是(1-1/m)，采m次还采不到，概率为 (1-1/m) 的m次方。
<div align=center>
<img src="/img/pics/2-5.png" />
</div>
那么我们可以计算得到，其实最终约有 36.8% 的样本是不在训练集中的。这种方法取出的测试样本在数据集D中比例一般在25%~36.8%之间。

> 它常用于数据集较小或难以有效划分训练/测试集情况。因为**数据分布有所改变，会引入估计误差**，初始数据量比较足够时，前两种方法更常用。

###  2.2 <a name='2-2'>性能度量</a>
**性能度量(performance measure)** 是衡量模型泛化能力的评价标准，反映了任务需求。使用不同的性能度量往往会导致不同的评判结果。

比如 回归(regression) 任务常用**均方误差**衡量性能：
<div align=center>
<img src="/img/pics/2-6.png" />
</div>

#### 2.2.1 <a name='2-2-1'>错误率与精度</a>
对于样例 D：
- **错误率（error rate）**：被错误分类的样本在总样本中的比例。
<div align=center>
<img src="/img/pics/2-7.png" />
</div>

- **精度（accuracy）**：被正确分类的样本在总样本中的比例，即（1 – 错误率）
<div align=center>
<img src="/img/pics/2-8.png" />
</div>

> 西瓜为例，**错误率是**有多少比例的瓜被判别错误。若关心“挑出的西瓜中有多少比例是好瓜”，或者“所有好瓜中有多少比例被挑了出来”，错误率就不够用了，需要使用其他性能度量。

> 有些需求在信息检索、Web搜索等应用中经常出现，例如在信息检索中，我们经常会关心“**检索出的信息中有多少比例是用户感兴趣的**”“**用户感兴趣的信息中有多少被检索出来了**”.**“ 查准率”(precision)与“查全率”(recal)是更为适用于此类需求的性能度量.**

#### 2.2.2 <a name='2-2-2'>查全率、查准率</a>

<div align=center>
<img src="/img/pics/2-9.png" />
</div>

- 解释：
    - FN: Fault Negative Example 错反例（错认为是反例）
    - FP: Fault Positive Example 错正例（错认为是正例）
    - 其他的反之亦然
- **查准率、准确率(Precision)**：预测为正的样例中有多少是真正的正样例（你认为的正例，有百分之多少是对的）
    - **P = TP/(TP+FP)**
- **查全率、召回率(Recall)**：样本中的正例有多少被预测正确（正例样本有百分之多少被你预测到了）
    - **R = TP/(TP+FN)**
- **精度（accuracy）**：被正确分类的样本在总样本中的比例（所有样本中预测正确的占百分之多少）
    - **A=(TP+TN)/(TP+FN+TN+FP)**

#### <a name='p-r'>**P-R曲线**</a>
- **问题**
    - 查准率和查全率是一对矛盾的度量
    - **查全率**是希望将好瓜尽可能多地挑选出来，所以选的瓜越多其实查全率也就越大。
    - **查准率**是选出的好瓜里面有多少是真正的好瓜，只选有把握的瓜查准率也就越大。
    - 但是这么看来，保证查准率的话，难免会漏掉好瓜，查全率得不到保证。
- **P-R曲线**
    - “P-R曲线”正是描述查准/查全率之间关系的曲线
    - **定义**：根据学习器的预测结果按正例可能性大小（与阈值之差）对样例进行排序，并逐个把样本作为正例计算P和R，得到很多个P和R，形成P-R图。

    <div align=center>
    <img src="/img/pics/2-10.png" />
    </div>
    
    - 一般来说，曲线下的面积是很难进行估算的，所以衍生出了 **“平衡点”（Break-Event Point，简称BEP）**，即当P=R时的取值，平衡点的取值越高，性能更优。
    - 当然 BEP 其实也太简化了一点，所以我们引入了新的度量方法：**F-Measure**。即 F 度量。
- **F-Measure**
    - 常用的是 **F1 度量**：(基于 P R 的调和平均值)
    <div align=center>
    <img src="/img/pics/2-11.png" />
    </div>
    
    - 然而有时候我们对查全率和查准率的要求有偏差，这个时候我们就需要修改它们的权重，所以又引入了**F<sub>β</sub> 度量**：（基于 P R 的加权平均值）
    <div align=center>
    <img src="/img/pics/2-12.png" />
    </div>
    
    > β > 1 时 R 查全率 有更大权重，β < 1 时 P 查准率 有更大权重。β = 1 时，则为 F1 度量。

#### <a name='m-m'>macro 宏观 vs micro 微观</a>
有时候我们会有多个二分类混淆矩阵，例如：多次训练或者在多个数据集上训练，那么估算全局性能的方法有两种，分为**宏观和微观**。

简单理解：
- **宏观**就是先算出每个混淆矩阵的P值和R值，然后取得平均P值macro-P和平均R值macro-R，再算出F<sub>β</sub>或F<sub>1</sub>，
- **微观**则是计算出混淆矩阵的平均TP、FP、TN、FN，接着进行计算P、R，进而求出F<sub>β</sub>或F<sub>1</sub>。

<div align=center>
<img src="/img/pics/2-13.png" />
</div>

#### <a name='r-a'>ROC 与 AOC</a>
如上所述：学习器对测试样本的**评估结果**一般为**一个实值或概率**，然后将这个预测值与一个 **分类阈值(threshold)** 比较，若大于阈值则分为正类，否则为反类，也就完成了分类。
> 学习器也就是分类器。

##### 概念
根据这个实值或概率预测结果，可将测试样本进行排序，“最可能”是正例的排在最前面，“最不可能”是正例的排在最后面。这种分类过程相当于在这个排序中以某个 **“截断点”(cut point)** 将样本分为两部分。根据任务需求采用**不同截断点**：更**重视“查准率”**，选择**靠前**的位置截断；更**重视“查全率”**，则从**靠后**的位置截断。
> 其实通俗来说的话，就是分类器会将测试样本评估，并给它们评分，然后按照评分排序，之后根据你侧重 P 还是 R 来设置一个截断点作为下次预测的判断依据。比如 你设置截断点 75分以上是 优秀，75分以下是 差，那就是我要求比较低；我设置 90分以上是 优秀，90分以下是 差，那就是我要求比较高。这是同一个道理。

因此，排序本身的质量好坏，就体现了 “一般情况下”泛化性能的好坏。

而 **ROC曲线** 正是从这个角度出发来研究学习器的泛化性能，ROC曲线与P-R曲线十分类似，都是按照排序的顺序逐一按照正例预测，不同的是ROC曲线以“真正例率”（True Positive Rate，简称TPR）为横轴，纵轴为“假正例率”（False Positive Rate，简称FPR），**ROC偏重研究基于测试样本评估值的排序好坏**。
> TPR = R ，TPR 和 FPR 可以看作 正例成功率 和 反例失败率，也就是 真正例TP占真正例的比例，假正例FP占真反例的比例。

<div align=center>
<img src="/img/pics/2-14.png" />
</div>
<div align=center>
<img src="/img/pics/2-15.png" />
</div>
简单分析图像，可以得知：当FN=0时，TN也必须0（TPR = FPR = 1），反之也成立，我们可以画一个队列，试着使用不同的截断点（即阈值）去分割队列，来分析曲线的形状，（0,0）表示将所有的样本预测为负例，（1,1）则表示将所有的样本预测为正例，（0,1）表示正例全部出现在负例之前的理想情况，（1,0）则表示负例全部出现在正例之前的最差情况。
<br>
<br>

##### 绘图
现实中的任务通常都是有限个测试样本，因此只能绘制出近似ROC曲线。绘制方法：首先根据测试样本的评估值对测试样本排序，接着按照以下规则进行绘制:

> 其实就是绘制 最大阈值时 的点 和 最小阈值时 的点，及其每个样本作为阈值的 点，构成一个离散点集，得到近似ROC曲线。
<div align=center>
<img src="/img/pics/2-16.png" />
</div>

##### 评估
同样地，进行模型的性能比较时，若一个学习器A的ROC曲线被另一个学习器B的ROC曲线**完全包住**（所有点更靠近（1，1）），则称B的性能优于A。

若A和B的曲线发生了交叉，则谁的曲线下的面积大，谁的性能更优。

所以就引出 ROC曲线下的面积 我们定义为**AUC（Area Uder ROC Curve）**，**不同于P-R的是，这里的AUC是可估算的**，即AOC曲线下每一个小矩形的面积之和。

<div align=center>
<img src="/img/pics/2-17.png" />
</div>

易知：AUC越大，证明排序的质量越好，AUC为1时，证明所有正例排在了负例的前面，AUC为0时，所有的负例排在了正例的前面，这两种情况其实都属于最理想的情况。（不过 AUC 为 0 肯定是 分类出问题了）

<div align=center>
<img src="/img/pics/2-18.png" />
</div>

#### 2.2.3 <a name='2-2-3'>代价敏感错误率与代价曲线</a>
上面的方法中，将学习器的犯错同等对待，但在现实生活中，将正例预测成假例与将假例预测成正例的代价常常是不一样的，例如：好瓜预测为坏瓜，顶多损失了好瓜；而坏瓜预测为好瓜，吃了会中毒，这是不一样的代价。

所以以二分类为例，由此引入了“代价矩阵”（cost matrix）。

<div align=center>
<img src="/img/pics/2-19.png" />
</div>

在非均等错误代价下，我们希望的是最小化“总体代价”，这样“代价敏感”的错误率（2.5.1节介绍）为：
<div align=center>
<img src="/img/pics/2-20.png" />
</div>
同样对于ROC曲线，在非均等错误代价下，演变成了“代价曲线”，在这里就不介绍了。

###  2.3 <a name='2-3'>比较检验</a>
在某种度量下取得评估结果后，我们是否可以直接比较以评判优劣？

不是这样的：
- 测试性能不等于泛化性能
- 测试性能随着测试集的变化而变化
- 很多机器学习算法本身有一定的随机性

我们的机器学习，也只是所谓的 “概率近似正确”，而不是真正的泛化。

我们可以根据数据集以及模型任务的特征，选择出最合适的评估和性能度量方法来计算出学习器的“测试误差”。但由于“测试误差”受到很多因素的影响，测试集的不同也会影响算法的评估，同时测试误差是作为泛化误差的近似，并不能代表学习器真实的泛化性能。所以我们提出了 比较检验 来对单个或多个学习器在不同或相同测试集上的性能度量结果做比较。

#### 2.3.1 <a name='2-3-1'>假设检验</a>
**“假设”指的是对样本总体的分布或已知分布中某个参数值的一种猜想**，例如：假设总体服从泊松分布，或假设正态总体的期望u=u0。回到本篇中，我们可以通过测试获得测试错误率，但直观上测试错误率和泛化错误率相差不会太远，因此可以通过测试错误率来推测泛化错误率的分布，**这就是一种假设检验**。

**统计假设检验(hypothesis test)** 为学习器性能比较提供了重要依据。

学习器的性能比较常用方法如下：
- #### 2.3.1.1 <a name='2-3-1-1'>**两学习器比较**</a>
    - 交叉验证t 检验(基于成对t 检验)
    <div align=center>
    <img src="/img/pics/2-21.png" />
    </div>

    > k 折交叉验证中训练集、测试集会产生重叠，可以通过5次2折交叉验证，使用第一次的两对差值计算均值，使用全部的差值对（即10对）计算方差，可以有效的避免这个问题。
    - McNemar 检验(基于列联表，卡方检验)
    > 主要思想是：若两学习器的性能相同，则A预测正确B预测错误数应等于B预测错误A预测正确数，即e01=e10，且|e01-e10|服从N（1，e01+e10）分布。
- #### 2.3.1.2 <a name='2-3-1-2'>**多学习器比较**</a>
    - Friedman + Nemenyi
        * Friedman检验(基于序值，F检验; 判断”是否都相同”)
        > F检验可以在多组数据集进行多个学习器性能的比较，基本思想是在同一组数据集上，根据测试结果（例：测试错误率）对学习器的性能进行排序，赋予序值1,2,3...，相同则平分序值.

        > 比如：以下是三个算法 ABC 在 四个数据集上的 序值。比如 D1 中，A 最好，B 其次，C 最差。
        <div align=center>
        <img src="/img/pics/2-22.png" />
        </div>
        
        > 若学习器的性能相同，则它们的平均序值应该相同，且第i个算法的平均序值ri服从正态分布N（（k+1）/2，（k+1）(k-1)/12），则有：

        <div align=center>
        <img src="/img/pics/2-23.png" />
        </div>
        <div align=center>
        <img src="/img/pics/2-24.png" />
        </div>
        * Nemenyi 后续检验(基于序值，进一步判断两两差别)
        
        > Friedman 检验检测出来所有算法都 “相同”，那么就它们的差别不显著，则需要进行 后续检验，Nemenyi 就是一个常用的方法。
        
        > Nemenyi检验计算出平均序值差别的临界值域，下表是常用的qa值，若两个算法的平均序值差超出了临界值域CD，则相应的置信度1-α拒绝“两个算法性能相同”的假设。
        <div align=center>
        <img src="/img/pics/2-25.png" />
        </div>
        <div align=center>
        <img src="/img/pics/2-26.png" />
        </div>

##  3 <a name='3'>偏差与方差</a>
在 [1.3 误差与精度](#1-3) 我们讲过误差，那么误差到底包含了哪些因素，或者说，如何从机器学习的角度来解释误差从何而来？

这里我们就要提到**偏差与方差**。**偏差-方差分解**是解释学习器泛化性能的重要工具。在学习算法中，偏差指的是预测的期望值与真实值的偏差，方差则是每一次预测值与预测值得期望之间的差均方。实际上，偏差体现了学习器预测的准确度，而方差体现了学习器预测的稳定性。
<div align=center>
<img src="/img/pics/2-27.png" />
</div>

但是 一般而言，偏差与方差存在冲突，这就是常说的偏差-方差窘境（bias-variance dilamma）：
- **训练不足**时，学习器拟合能力不强，偏差主导了泛化错误率（欠拟合）
- 随着**训练程度加深**，学习器拟合能力逐渐增强，方差逐 渐主导了泛化错误率
- **训练充足**后，学习器的拟合能力很强，方差主导了泛化错误率（过拟合）
<div align=center>
<img src="/img/pics/2-28.png" />
</div>
这也就是绪论所说的拟合的几个情况。