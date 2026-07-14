# Datawhale 零基础入门数据挖掘 Baseline 笔记

## 介绍
目标：

> 根据二手车的历史交易数据，预测车辆的交易价格（price）。
**监督学习回归问题**。
数据：

- 训练集：
  - 样本数量：150000
  - 特征数量：30
  - 包含目标变量 `price`

- 测试集：
  - 样本数量：50000
  - 特征数量：29
  - 不包含 `price`

评价指标：

- MAE（Mean Absolute Error，平均绝对误差）

公式：

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
$$


---

# 导入工具库


## 基础工具

```python
import numpy as np
import pandas as pd

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

%matplotlib inline
````

## 模型相关

```python
from sklearn import linear_model
from sklearn import preprocessing

from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

import lightgbm as lgb
import xgboost as xgb
```

## 降维工具

```python
from sklearn.decomposition import (
    PCA,
    FastICA,
    FactorAnalysis,
    SparsePCA
)
```

## 模型评价与参数搜索

```python
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
    train_test_split
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)
```

---

#  数据读取


查看数据规模：

```python
print('Train data shape:', Train_data.shape)

print('TestA data shape:', TestA_data.shape)
```

---

# 1. 数据探索分析 EDA

## 查看前5行

```python
Train_data.head()
```


---

## 查看数据类型

```python
Train_data.info()
```


数据包含：

* 数值特征
* 类别特征
* 缺失值

---

## 查看字段名称

```python
Train_data.columns
```

---

## 查看统计信息

```python
Train_data.describe()
```

可以查看：

* 均值
* 方差
* 最大值
* 最小值
* 四分位数


---

# Step 3 特征工程

## 1. 查看数值特征

```python
numerical_cols = Train_data.select_dtypes(
    exclude='object'
).columns
```

结果：

包含：

* int
* float

---

## 2. 查看类别特征

```python
categorical_cols = Train_data.select_dtypes(
    include='object'
).columns
```

结果：

```text
notRepairedDamage
```

---

# 构建训练数据

## 特征选择

删除：

* SaleID
* name
* regDate
* creatDate
* price
* model
* brand
* regionCode
* seller

原因：

部分字段：

* ID无实际意义
* 高基数类别容易过拟合
* 日期需要进一步处理

代码：

```python
feature_cols = [
    col for col in numerical_cols
    if col not in [
        'SaleID',
        'name',
        'regDate',
        'creatDate',
        'price',
        'model',
        'brand',
        'regionCode',
        'seller'
    ]
]


feature_cols = [
    col for col in feature_cols
    if 'Type' not in col
]
```

---

## 构造训练集和测试集

```python
X_data = Train_data[feature_cols]

Y_data = Train_data['price']


X_test = TestA_data[feature_cols]
```

数据：

```
X train shape:(150000,18)

X test shape:(50000,18)
```

---

# 标签分析

定义统计函数：

```python
def Sta_inf(data):

    print('_min',np.min(data))

    print('_max:',np.max(data))

    print('_mean',np.mean(data))

    print('_ptp',np.ptp(data))

    print('_std',np.std(data))

    print('_var',np.var(data))
```

查看price：

```python
Sta_inf(Y_data)
```


---

## 查看价格分布

```python
plt.hist(Y_data)

plt.show()
```

发现：

价格呈现明显长尾分布。

---

# 缺失值处理

直接填充：

```python
X_data = X_data.fillna(-1)

X_test = X_test.fillna(-1)
```

原因：

树模型能够处理离散异常值。

---

# 模型训练

本Baseline使用：

* XGBoost
* LightGBM

两个GBDT模型融合。

---

# 1. XGBoost模型

参数：

```python
xgr = xgb.XGBRegressor(

    n_estimators=120,

    learning_rate=0.1,

    gamma=0,

    subsample=0.8,

    colsample_bytree=0.9,

    max_depth=7
)
```

参数解释：

| 参数               | 作用     |
| ---------------- | ------ |
| n_estimators     | 树数量    |
| learning_rate    | 学习率    |
| max_depth        | 树深度    |
| subsample        | 样本采样比例 |
| colsample_bytree | 特征采样比例 |

---

# 2. 五折交叉验证

使用：

```python
StratifiedKFold
```

划分：

```
训练集80%

验证集20%
```

训练：

```python
for train_ind,val_ind in sk.split(
    X_data,
    Y_data
):

    xgr.fit(
        train_x,
        train_y
    )
```

评价：

```python
mean_absolute_error()
```

结果：

```
Train MAE:

628


Validation MAE:

715
```

---

# 3. LightGBM模型

模型：

```python
lgb.LGBMRegressor(
    num_leaves=127,
    n_estimators=150
)
```

参数搜索：

```python
param_grid={

'learning_rate':
[
0.01,
0.05,0.1,0.2
]

}
```

使用：

```python
GridSearchCV
```

寻找最佳学习率。

---

# 数据划分

```python
x_train,x_val,y_train,y_val = train_test_split(

    X_data,

    Y_data,

    test_size=0.3

)
```

比例：

```
训练70%

验证30%
```

---

# LightGBM训练

```python
model_lgb = build_model_lgb(
    x_train,
    y_train
)
```

验证：

```python
MAE_lgb =
mean_absolute_error(
    y_val,
    val_lgb
)
```


# XGBoost训练

```python
model_xgb = build_model_xgb(
    x_train,
    y_train
)
```

---

# Step 5 模型融合

## 为什么融合？

不同模型学习能力不同：

* LightGBM：

  * 叶子节点生长策略
  * 对类别特征敏感

* XGBoost：

  * 正则能力强
  * 泛化能力较好

融合可以降低误差。

---

## 加权融合

公式：

$$
Prediction
==========

w_1LGB+w_2XGB
$$

权重：

根据MAE动态计算。

代码：

```python
val_Weighted = (

    1-MAE_lgb/(MAE_xgb+MAE_lgb)

)*val_lgb + (

    1-MAE_xgb/(MAE_xgb+MAE_lgb)

)*val_xgb
```

修正负价格：

```python
val_Weighted[val_Weighted<0]=10
```


相比：

| 模型       | MAE |
| -------- | --- |
| XGBoost  | 715 |
| LightGBM | 689 |
| 融合       | 687 |

融合效果最好。

---

# Step 6 生成提交文件

预测测试集：

```python
sub_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*subA_lgb + (1-MAE_xgb/(MAE_xgb+MAE_lgb))*subA_xgb
```

创建提交文件：

```python
sub=pd.DataFrame()

sub['SaleID']=TestA_data.SaleID

sub['price']=sub_Weighted


sub.to_csv(
    './sub_Weighted.csv',
    index=False
)
```

提交格式：

| SaleID | price   |
| ------ | ------- |
| 0      | 39533.7 |
| 1      | 386.0   |
| 2      | 7791.9  |

---



## 完整比赛流程

```
数据读取

↓

EDA数据探索

↓

特征工程

↓

缺失值处理

↓

划分训练验证集

↓

训练模型

↓

模型评价

↓

模型融合

↓

生成提交文件
```

---

# Baseline方法总结

| 步骤   | 方法       |
| ---- | -------- |
| 数据处理 | Pandas   |
| 缺失值  | 填充-1     |
| 模型1  | XGBoost  |
| 模型2  | LightGBM |
| 验证   | MAE      |
| 融合   | 加权平均     |
| 任务类型 | 回归预测     |

---

# 后续优化方向

## 特征工程

可以增加：

* 注册年份
* 使用年限

例如：

$$
car_age
=======

saleYear-regYear
$$

## 特征编码

增加：

* brand平均价格
* model平均价格
* region价格统计

## 模型优化

尝试：

* CatBoost
* LightGBM调参
* XGBoost调参
* Stacking

## 数据处理

尝试：

* price log变换
* 异常值处理
* 特征归一化

该Baseline完成了一个完整的数据挖掘竞赛流程。

```
```
