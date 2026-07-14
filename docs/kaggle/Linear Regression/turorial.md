# Linear Regression With Time Series

## 定义：What is a Time Series?

预测的基本对象是时间序列，它是一组随时间记录的观测值。在预测应用中，观测值通常是按固定频率记录的，比如每天或每月。

### 时间序列线性回归

最小二乘法

**时间步特征（time-step features）**

时间序列有两种独特的特征：时间步特征和滞后特征。

*时间步特征*是直接从时间戳（Timestamp）中提取出来的特征。

| 特征 | 含义 |
|------------|------|
| Year | 年 |
| Month | 月 |
| Day | 日 |
| Hour | 小时 |
| Minute | 分钟 |
| Weekday | 星期几 |
| Week of Year | 第几周 |
| Quarter | 第几季度 |
| Is Weekend | 是否周末 |
| Holiday | 是否节假日 |
| Season | 春夏秋冬 |

*滞后特征*把过去的数据作为现在的输入。

```python
df = tunnel.copy()
df['time'] = np.arange(len(tunnel.index))
df.head()

from sklearn.linear_model import LinearRegression

X = df.loc[:, ['time']]
y = df.loc[:, 'NumVehicles']

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

df['Lag_1'] = df['NumVehicles'].shift(1)
df.head()

from sklearn.linear_model import LinearRegession

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=Ture)
y = df.loc[:, 'NumVehecles']
y, X = y.align(X, join='inner')

model = LinearRegresion()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
```
## What is Trend?
 时间序列的趋势成分表示序列均值的持续、长期变化。趋势是序列中最缓慢变化的部分，代表最重要的大时间尺度。在一个产品销售的时间序列中，上升趋势可能是由于市场扩展，随着每年越来越多的人了解该产品而产生的效果。

# 移动平均图
 通过在某个定义好的窗口宽度内计算值的平均值来实现

## 完整代码
'''python
moving_average = tunnel.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()

ax = tunnel.plot(style=".", color="0.5")

moving_average.plot(
    ax = ax,
    linewidth=3,
    title="Tunnel Traffic - 354-Day Moving Averge",
    legend=False,
)
'''

---

# 什么是moving average？
 > **对于每一天，不直接看当天的数据，而是看他付近一段时间的平均值。**

 rolling()
 '''python
 tunnel.ronlling()
 '''

> **建立一个可以会移动的窗口**

window 控制：

> **一次统计多少个数据。**

## DeterministicProcess

'DeterministicProcess' 是 'statsmodels' 提供的一个**确定性时间特征生成器**，可以根据时间索引自动生存时间特征，例如时间步(Trend)、多项式趋势等，而不需要手动创建'range(len(df))

'''python
from statsdmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,
    costant=True,
    order=1,
    drop=True,
)
'''

### 参数说明

 | 参数 | 作用 |
 |------|------|
 |'index=tunnel.index'|使用数据的时间索引生成特征，保证生成的数据和原数据一一对应|
 |'constant=True'|添加一列全为'1' 的常数列，作为线性回归的截距|
 |'order=1'| 生成一节趋势，相当于'time=range(len(df))'|
 |'drop=True'|自动删除共线特征，避免多重共线性|



 ### 生成训练特征

```python
X = dp.in_sample()
```

生成结果类似于：

| Date | const | trend |
|------|------:|------:|
|2003-01-01|1|1|
|2003-01-02|1|2|
|2003-01-03|1|3|

随后即可用于训练模型：

```python
model.fit(X, y)
```

### 与手动创建 Time Feature 的区别

之前课程中我们使用：

```python
df["time"] = range(len(df))
```

只会生成一个时间步特征。

`DeterministicProcess` 则可以统一生成更多时间特征，例如：

- Trend（时间步）
- Trend²、Trend³（多项式趋势）
- Seasonality（季节性）
- Fourier 特征（周期特征）

因此在时间序列建模中更加规范，也便于后续生成未来时间点的特征。