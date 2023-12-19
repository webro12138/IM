import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from diffusionModel import ICWeighter, IC
from algorithm import DegreeDiscount
from dataset import load_network, load_features
import networkx as nx
import numpy as np


network = load_network("stelzl", "undirected")

## 定义权重生成器
weighter = ICWeighter(APType="random", p=(0, 0.5))

## 定义IC模型
ic = IC(weighter=weighter, MC=1000, verbose=True)



## 定义DegreeDiscount算法
degree_discount = DegreeDiscount()


S = degree_discount(network, 20)
print(ic(network, S))


