---
title: 常见算法
date: 2024-09-28 22:01:52
tags: [DL-notes,DL,notes]
categories: [深度学习,笔记]
---



# 算法



# 常见算法

## 常见的激活函数

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha):
    return np.where(x>0, x, x*alpha)

def elu(x, alpha):
    return np.where(x>0, x, alpha*(np.exp(x) - 1))
```



## 梯度下降

```python
lr = 0.1
x1 = 2.5

for i in range(n_iterations):
    gredient = df(x1)
    x1 = x1 - lr*gredient
```



## 概率分布模型

```python
import numpy as np
# 设定随机数种子，以便于复现结果
np.random.seed(0)

# 伯努利
p = 0.5
bernoulli_dist = np.random.binomial(1, p, 1000)

# 二项
n = 10
binomial_dist = np.random.binomial(n, p, 1000)

# 正态
mu, sigma= 0, 0.1
normal_dist = np.random.normal(mu, sigma, 1000)

# 指数
lambd = 1.0 
exponential_dist = np. random.exponential(1/lambd, 1000)

# Logistics
mu, s = 0, 1
logistic_dist = np.random.logistic(mu, s, 1000)
```

```python
import numpy as np

data1 = np.random.randn(1000)
data2 = np.random.randn(1000)
# 期望
expectation1 = np.mean(data1)
expectation2 = np.mean(data2)
# 方差
variance1 = np.var(data1)
variance2 = np.var(data2)
# 协方差
covariance_matrix = np.cov(data1, data2)
```





## 逻辑回归模型





## 反向传播算法

```python
import numpy as np

x_1 = 40.0
x_2 = 80.0

expected_output = 60.0

# 初始化

# 网络权重
w_1 = np.full((2,3),0.5)
w_2 = np.full((3,1),1.0)

def back_forward(x_1, x_2, w_1, w_2, expected_output, loop_num, print_num):
    for i in range(1,loop_num):
        z_1 = x_1 * w_1[0][0] + x_2 * w_1[1][0]
        z_2 = x_1 * w_1[0][1] + x_2 * w_1[1][1]
        z_3 = x_1 * w_1[0][2] + x_2 * w_1[1][2]

        y_pred = z_1 * w_2[0][0] + z_2 * w_2[1][0] + z_3 * w_2[2][0]
        loss = 0.5 * (expected_output - y_pred) ** 2

        if(i%print_num == 0):
            print(f"前向结果：{y_pred}")    
            print(f"Loss：{loss}")

        # 计算梯度
        # 输出层梯度
        d_loss_predict_output = -(expected_output - y_pred)

        # 权重损失梯度
        d_loss_w_2 = np.multiply(d_loss_predict_output,[z_1, z_2, z_3])
        d_loss_w_1 = np.multiply(np.multiply(w_2,[x_1,x_2]),d_loss_predict_output)

        # d_loss_w_1[0][0] -> w_2[0][0] * x_1 

        learning_rate = 1e-5
        w_2 -= np.multiply(d_loss_w_2, learning_rate).reshape(3,1)

        w_1 -= np.multiply(d_loss_w_1, learning_rate).reshape(2,3)

        # 继续前向

print(f"expected_output：{expected_output}")
back_forward(x_1, x_2, w_1, w_2, expected_output, 100, 10)

```

