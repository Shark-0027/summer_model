# Datawhale 零基础入门数据挖掘 Task2：EDA 数据探索性分析

## 1. EDA（Exploratory Data Analysis）数据探索性分析

EDA（探索性数据分析）是机器学习建模前的重要步骤，主要目标是通过统计分析和可视化方法了解数据集的结构、分布、异常情况以及变量之间的关系，为后续的数据清洗、特征工程和模型训练提供依据。

在二手车交易价格预测任务中，EDA主要用于：

- 熟悉训练集和测试集的数据结构；
- 分析特征类型以及特征含义；
- 检查缺失值和异常值；
- 分析预测目标 price 的分布情况；
- 探索特征与价格之间的相关关系；
- 为后续特征工程提供方向。

---

# 2. EDA流程

完整EDA流程：

1. 导入数据分析和可视化库；
2. 加载训练集和测试集；
3. 查看数据基本信息；
4. 分析缺失值和异常值；
5. 分析目标变量分布；
6. 区分类别特征和数值特征；
7. 数值特征分析；
8. 类别特征分析；
9. 自动生成数据分析报告。

---

# 3. 导入数据分析库

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
````

常用库：

| 库          | 作用         |
| ---------- | ---------- |
| pandas     | 数据读取、清洗、分析 |
| numpy      | 数值计算       |
| matplotlib | 基础绘图       |
| seaborn    | 高级统计可视化    |
| missingno  | 缺失值可视化     |

---


## 5.3 匿名特征

数据中存在：

```text
v_0 ~ v_14
```

共15个匿名特征。

由于进行了脱敏处理，无法知道具体业务含义，但可以通过：

* 分布分析；
* 相关性分析；
* 模型特征重要性；

判断其预测能力。

---

# 6. 查看数据基本情况

## 6.1 查看前后几行

```python
Train_data.head().append(
    Train_data.tail()
)
```

作用：

* 查看数据格式；
* 判断字段是否正确读取；
* 观察数据是否存在明显异常。

---


## 经验：

在数据分析过程中，应养成：

```python
head()
shape
info()
describe()
```

四个函数结合查看数据的习惯。

每完成一步数据处理，都建议重新查看结果，避免后续出现连锁错误。

---

# 7. 数据总体统计分析

## 7.1 describe()

```python
Train_data.describe()
```

describe() 可以查看：

* count：非空数量；
* mean：平均值；
* std：标准差；
* min：最小值；
* 25%：四分之一分位数；
* 50%：中位数；
* 75%：四分之三分位数；
* max：最大值。

---


### 1. 数据范围

例如：

```text
power 最大值 = 19312
```

而正常汽车功率不会达到这个数值。

说明：

可能存在异常值。

---

### 2. 缺失值

例如：

```text
bodyType count < 总样本数量
```

说明存在缺失。

---

### 3. 特殊异常编码

部分数据可能使用：

```text
999
9999
-1
```

表示缺失值。

EDA阶段需要重点检查。

---


## 类型意义

### 数值类型

例如：

```text
power
kilometer
v_0
v_1
```

可以直接进行：

* 相关性分析；
* 分布分析；
* 回归分析。

### object类型

例如：

```text
notRepairedDamage
```

需要进一步检查。

---

# 9. 缺失值分析

## 9.1 查看缺失数量

```python
Train_data.isnull().sum()
```

结果：

主要缺失字段：

| 字段       | 缺失数量 |
| -------- | ---: |
| model    |    1 |
| bodyType | 4506 |
| fuelType | 8680 |
| gearbox  | 5981 |

---

## 9.2 缺失值可视化

```python
missing = Train_data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
```

作用：

直观看出：

* 哪些字段缺失；
* 缺失比例大小。

---

## 9.3 missingno可视化

```python
msno.matrix(
    Train_data.sample(250)
)
```

查看缺失值分布。

```python
msno.bar(
    Train_data.sample(1000)
)
```

查看每列有效数据比例。

---

# 缺失值处理原则

对于缺失值：

## 方法1：填充

例如：

均值填充：

```python
mean()
```

众数填充：

```python
mode()
```

## 方法2：模型自动处理

例如：

LightGBM

可以直接利用：

```text
NaN
```

进行分裂优化。

## 方法3：删除

如果：

* 缺失比例过高；
* 特征价值较低；

可以删除。

---

# 10. 异常值分析

## 检查object字段

```python
Train_data['notRepairedDamage'].value_counts()
```

发现：

```text
0.0
1.0
-
```

其中：

```text
-
```

不是正常类别，而是缺失值。

---

转换：

```python
Train_data[
    'notRepairedDamage'
].replace(
    '-',
    np.nan,
    inplace=True
)
```

处理后：

```text
0.0
1.0
NaN
```

更加符合机器学习处理方式。

---

# 11. 删除无意义特征

查看：

```python
Train_data["seller"].value_counts()

Train_data["offerType"].value_counts()
```

发现：

seller：

```text
0 : 149999
1 : 1
```

offerType：

```text
0 : 150000
```

说明：

两个字段几乎没有区分能力。

删除：

```python
del Train_data["seller"]
del Train_data["offerType"]

del Test_data["seller"]
del Test_data["offerType"]
```

---

# 总结

本节完成了数据探索的基础流程：

* 加载数据；
* 查看数据结构；
* 分析字段类型；
* 检测缺失值；
* 发现异常编码；
* 删除无效特征。

EDA的核心目标不是直接建模，而是：

> 充分理解数据，让后续特征工程和模型训练更加可靠。

```

````markdown id="48291"
# 12. 目标变量 price 分布分析

在机器学习回归任务中，目标变量（label）的分布情况会直接影响模型训练效果。因此，在建模之前需要分析 `price` 的：

- 整体分布；
- 偏度（Skewness）；
- 峰度（Kurtosis）；
- 是否存在异常值；
- 是否需要进行数据变换。

---

# 12.1 查看目标变量

```python
Train_data['price']
````

`price` 是二手车交易价格，也是模型最终需要预测的目标。

---

# 12.2 查看价格频数分布

```python
Train_data['price'].value_counts()
```

通过统计每个价格出现次数，可以观察：

* 哪些价格出现频率较高；
* 是否存在极端价格；
* 是否存在异常样本。

例如：

```text
500
1500
1200
1000
2500
```

这些价格出现次数较多。

而：

```text
40000+
```

价格样本数量较少。

说明：

二手车价格数据存在明显长尾分布。

---

# 12.3 查看价格分布情况

## Johnson SU分布

```python
import scipy.stats as st

y = Train_data['price']

plt.figure(1)
plt.title('Johnson SU')

sns.distplot(
    y,
    kde=False,
    fit=st.johnsonsu
)
```

---

## 正态分布

```python
plt.figure(2)
plt.title('Normal')

sns.distplot(
    y,
    kde=False,
    fit=st.norm
)
```

---

## 对数正态分布

```python
plt.figure(3)
plt.title('Log Normal')

sns.distplot(
    y,
    kde=False,
    fit=st.lognorm
)
```

---

分析发现：

`price` 并不服从标准正态分布。

价格数据具有：

* 右偏；
* 长尾；
* 极端高价格样本。

因此，在回归模型训练中，可以考虑：

```text
log变换
```

降低数据偏态。

---

# 12.4 偏度 Skewness 和峰度 Kurtosis

## 偏度

Skewness 用来描述数据分布的不对称程度。

公式：

$$
Skewness=\frac{E(X-\mu)^3}{\sigma^3}
$$

判断：

| 偏度 | 含义   |
| -- | ---- |
| 0  | 完全对称 |
| >0 | 右偏   |
| <0 | 左偏   |

---

## 峰度

Kurtosis 用于描述数据尾部厚度。

公式：

$$
Kurtosis=\frac{E(X-\mu)^4}{\sigma^4}
$$

判断：

| 峰度 | 含义     |
| -- | ------ |
| 较高 | 存在极端值  |
| 较低 | 分布更加平坦 |

---

# 12.5 计算price偏度和峰度

```python
sns.distplot(
    Train_data['price']
)

print(
    "Skewness: %f"
    %
    Train_data['price'].skew()
)

print(
    "Kurtosis: %f"
    %
    Train_data['price'].kurt()
)
```

结果：

```text
Skewness: 3.346487

Kurtosis: 18.995183
```

说明：

## 偏度

```text
3.34
```

价格明显右偏。

即：

少量高价格车辆拉高整体分布。

## 峰度

```text
18.99
```

峰度较高。

说明：

存在较多极端价格。

---

# 12.6 所有特征偏度分析

```python
Train_data.skew()
```

可以查看所有字段的偏度。

例如：

## power

```text
65.86
```

说明：

汽车功率存在严重异常值。

---

## creatDate

```text
-79
```

说明：

日期字段可能需要进一步处理。

---

可视化：

```python
sns.distplot(
    Train_data.skew(),
    color='blue',
    axlabel='Skewness'
)
```

查看整体特征偏度分布。

---

峰度：

```python
sns.distplot(
    Train_data.kurt(),
    color='orange',
    axlabel='Kurtosis'
)
```

查看整体峰度情况。

---

# 12.7 价格频数可视化

```python
plt.hist(
    Train_data['price'],
    orientation='vertical',
    histtype='bar',
    color='red'
)

plt.show()
```

观察发现：

* 大部分车辆价格集中在低价区间；
* 高价格车辆数量很少；
* 存在明显长尾。

---

# 12.8 对price进行log变换

由于价格分布不均匀，可以进行：

$$
y'=log(1+y)
$$

代码：

```python
plt.hist(
    np.log(Train_data['price']),
    orientation='vertical',
    histtype='bar'
)

plt.show()
```

作用：

* 缩小极端值影响；
* 降低数据偏度；
* 提升回归模型稳定性。

---

# 13. 特征类型划分

机器学习中通常需要区分：

* 数值特征（Numerical Feature）
* 类别特征（Categorical Feature）

不同类型特征需要不同分析方式。

---

# 13.1 提取预测目标

```python
Y_train = Train_data['price']
```

保存标签。

---

# 13.2 数值特征

本任务中的数值特征：

```python
numeric_features = [
    'power',
    'kilometer',
    'v_0',
    'v_1',
    'v_2',
    'v_3',
    'v_4',
    'v_5',
    'v_6',
    'v_7',
    'v_8',
    'v_9',
    'v_10',
    'v_11',
    'v_12',
    'v_13',
    'v_14'
]
```

主要包括：

* 汽车功率；
* 行驶公里数；
* 匿名连续变量。

---

# 13.3 类别特征

```python
categorical_features = [
    'name',
    'model',
    'brand',
    'bodyType',
    'fuelType',
    'gearbox',
    'notRepairedDamage',
    'regionCode'
]
```

类别特征包括：

* 品牌；
* 车型；
* 车身类型；
* 燃油类型；
* 变速箱；
* 地区编码。

---

# 14. 类别特征唯一值分析

## 查看每个类别数量

```python
for cat_fea in categorical_features:

    print(
        cat_fea
        +
        "特征分布如下:"
    )

    print(
        Train_data[cat_fea].nunique()
    )

    print(
        Train_data[cat_fea].value_counts()
    )
```

---

# 分析结果

## name

```text
99662
```

类别数量非常多。

说明：

属于高基数类别。

后续处理需要注意：

* target encoding；
* frequency encoding；
* embedding。

---

## model

```text
248
```

车型数量较多。

可能与价格强相关。

---

## brand

```text
40
```

品牌数量较少。

适合：

* One-Hot Encoding；
* Label Encoding。

---

## bodyType

```text
8
```

车身类型较少。

---

## fuelType

```text
7
```

燃油类型较少。

---

## gearbox

```text
2
```

二分类：

```text
自动挡
手动挡
```

---

## regionCode

```text
7905
```

地区编码数量非常多。

属于高维类别变量。

需要谨慎处理。

---

# 15. 数值特征分析

数值特征主要分析：

1. 特征与price相关性；
2. 偏度和峰度；
3. 单变量分布；
4. 多变量关系。

---

# 15.1 相关性分析

相关系数：

$$
r=
\frac{cov(X,Y)}
{\sigma_X\sigma_Y}
$$

范围：

$$
[-1,1]
$$

含义：

| 范围   | 关系    |
| ---- | ----- |
| 接近1  | 强正相关  |
| 接近-1 | 强负相关  |
| 接近0  | 无明显关系 |

---

代码：

```python
price_numeric = Train_data[numeric_features]

correlation = price_numeric.corr()

print(
    correlation['price']
    .sort_values(
        ascending=False
    )
)
```

---

结果：

与price相关性较高：

| 特征        |  相关系数 |
| --------- | ----: |
| v_12      |  0.69 |
| v_8       |  0.68 |
| v_0       |  0.62 |
| power     |  0.21 |
| kilometer | -0.44 |
| v_3       | -0.73 |

---

分析：

## 正相关

例如：

```text
v_12
v_8
v_0
```

数值增加：

price倾向增加。

---

## 负相关

例如：

```text
v_3
kilometer
```

说明：

行驶里程越高：

车辆价格通常越低。

符合业务逻辑。

---

# 15.2 相关性热力图

```python
f, ax = plt.subplots(
    figsize=(7,7)
)

sns.heatmap(
    correlation,
    square=True,
    vmax=0.8
)
```

作用：

直观看：

* 特征之间相关关系；
* 特征与目标变量关系。

---

# 小结

目标变量分析阶段主要完成：

* price分布分析；
* 判断数据偏态；
* 判断异常值；
* log变换思路；
* 特征类型划分；
* 类别特征基数分析；
* 数值特征相关性分析。

这些结果会直接指导后续：

* 数据清洗；
* 特征工程；
* 模型选择。

```


