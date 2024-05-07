---

title: 单细胞数据分析
date: 2024-04-24 16:55:38
tags: [single-cell,人工智能]
---

# 单细胞数据分析

自2013年被选为年度方法以来，单细胞技术已经足够成熟，可以为复杂的研究问题提供答案。随着单细胞分析技术的发展，从单细胞分析中收集的数据也显著增加，导致处理这些庞大而复杂的数据集的计算挑战。



## 单细胞RNA测序-数据分析

scanpy 是一个用于分析单细胞转录组(single cell rna sequencing) 数据的python库，文章2018发表在[*Genome Biology*](https://genomebiology.biomedcentral.com/)。它和seurat几乎大差不差，但是以Python的生态，完全可以认为其具有更大的扩展潜力。

### 安装环境（[Scanpy单细胞测序学习-环境配置](https://yingbio.cn/archives/scseq-scanpy-install)）

### 公共单细胞数据集

10X Genomics免费提供的外周血单核细胞(PBMC)数据集

[Preprocessing and clustering 3k PBMCs (legacy workflow) — scanpy-tutorials 0.1.dev50+g06018e6 documentation](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)

scanpy提供的公开数据集

[Datasets — scanpy](https://scanpy.readthedocs.io/en/stable/api/datasets.html)

### 开始

#### 1. 载入包

```python
# 载入包
import numpy as np
import pandas as pd
import scanpy as sc

# 设置日志等级: errors (0), warnings (1), info (2), hints (3)
sc.settings.verbosity = 3             
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

> scanpy==1.10.1 anndata==0.10.7 umap==0.5.5 numpy==1.23.1 scipy==1.13.0 pandas==2.2.2 scikit-learn==1.4.2 statsmodels==0.14.1 igraph==0.11.4 pynndescent==0.5.12

#### 2. 载入数据集

```python
# sc载入数据集
adata = sc.datasets.pbmc3k()
# 本地载入数据集
results_file = 'D:\scanpy\write\pbmc3k.h5ad' 
help(sc.read_10x_mtx)
adata = sc.read_10x_mtx(
    'D:/scanpy/data/filtered_feature_bc_matrix',  # the directory with the `.mtx` file
    var_names='gene_symbols',                  # use gene symbols for the variable names (variables-axis index)
    cache=True) 
adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
```

#### 3. 查看数据

```python
adata
```

<img src="D:\code\SapientialM\source\_posts\single-cell\assets\image-20240423155809136.png" alt="image-20240423155809136" style="zoom:67%;" />

AnnData object with n_obs × n_vars = 2700 × 32738 意思是这是一个AnnData对象，n_obs 即有2700个细胞样本，n_vars 即有32738个基因序列。



##### 数据对象 （[anndata - Annotated data](https://anndata.readthedocs.io/en/latest/index.html)）

Scanpy 构建的对象叫做 AnnData 对象：

标准的AnnData对象主要包括以下几个部分（如果你要调用这些属性，只需要直接添加到后面即可，比如adata.obs、adata.X等）：

- .X: 存储基因表达矩阵，行代表基因，列代表细胞，也就是**显示的 n_obs x n_vars**。
- .obs: **观测值数据**，存储细胞相关的注释信息，例如细胞类型、样本信息等。
- .var: **特征和高可变数据**，存储基因相关的注释信息，例如基因名称、基因类型等。
- .uns: **非结构化数据**，存储与数据分析相关的信息，例如数据预处理参数、可视化参数等；可以包含一些在分析数据时我们得到的一些有价值的信息。
- .obsm: **细胞的附加数据**，例如细胞的空间位置、转录组拆分信息等；也就是我们进行进一步处理后得到的细胞数据。
- .varm: **基因的附加数据**，例如基因的表达模式、变异信息等；也就是我们处理得到的基因级别的元数据。
- .layers: **各种类型的基因表达矩阵**，例如原始表达矩阵、归一化表达矩阵等；我们可能拥有不同形式的原始核心数据，也许一种是规范化的，或者不是。这些可以存储在 AnnData 的不同layer中。
- varp、obsp: **基因、细胞映射关系的附加数据**，我没有在相关文档找到varp的详细说明，但应该类似于varm、obsm，拥有 n_obs x n_obs 和 n_var x n_var 大小的矩阵，用于数据分析国产中得到的一些映射关系的信息，比如A基因与B基因之间的关系，A细胞与B细胞之间的关系。

> p即Pairwise annotation，m即Multi-dimensional annotation，obs即observations，var即variables/ features，uns即Unstructured annotation，具体的数据解释可参考：[anndata.AnnData — anndata 0.1.dev50+g0a768fc documentation](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html)

整个AnnData对象如下所示结构，具体意思就是，根据var即基因信息作为属性列包含了 var 表、X表、varm表、varp表，同样的obs作为属性列包含了obs表、X表、obsm表、obsp表

<img src="https://anndata.readthedocs.io/en/latest/_static/anndata_schema.svg" width="50%" height="50%">

##### 数据核心

单细胞转录组的核心就是一个cell X gene的二维表，

#### 4. 查看数据





### 参考资料

推荐一本书：

[《利用 Python 进行数据分析 · 第 2 版》  · BookStack](https://www.bookstack.cn/read/pyda-2e-zh/README.md)

参考资料：

[Single Cell data analysis tutorial on PBMC dataset using scanpy - Part1](https://www.youtube.com/watch?v=_tP6vCwZfuY)

[Youtube/scanpy/PBMC_data at main · ramadatta/Youtube (github.com)](https://github.com/ramadatta/Youtube/tree/main/scanpy/PBMC_data)

[【基于python的单细胞分析】使用scVI实现批次效应校正](https://mp.weixin.qq.com/s/c_3NjoJyZkSv1XjIYu-V_g)

[【基于python的单细胞分析】如何进行细胞类型注释](https://mp.weixin.qq.com/s/ekJ0gyMqnchx5_6U4WoPHQ)

[基于Scanpy的单细胞数据质控、聚类、标注](https://mp.weixin.qq.com/s/u5fkFnTe_eDe1F2RTkdtLQ)

[**scanpy 单细胞分析包图文详解 01 | 深入理解 AnnData 数据结构**](https://blog.51cto.com/u_14782715/5082964)

[【陈巍学基因】单细胞RNA测序分析图解读](https://www.youtube.com/watch?v=NYpwinpPEb0)

[Scanpy单细胞测序学习-环境配置](https://yingbio.cn/archives/scseq-scanpy-install)

[基于COSG的单细胞数据marker基因鉴定](https://mp.weixin.qq.com/s/IlG2R7qXCHpOH94cQHRRdA)

[scanpy教程：预处理与聚类-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1610396)

[预处理和聚类 3k PBMC（旧工作流） — scanpy-tutorials 0.1.dev50+g06018e6 文档](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
