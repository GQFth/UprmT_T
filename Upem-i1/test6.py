# RMSNorm
# 公式通过 均方根缩放 + 科学系增益 实现归一化

import numpy as np
import random

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.eps = eps
        self.weight = np.ones(dim)
    
    def __call__(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1,keepdims=True) + self.eps)
        normed = x / rms
        return normed * self.weight

x = np.random.randn(2, 4)
norm = RMSNorm(4)
y = norm(x)
print("输入：", x)
print("输出：", y)