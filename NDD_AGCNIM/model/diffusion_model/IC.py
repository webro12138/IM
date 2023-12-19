import networkx as nx
import random
import numpy as np
# 基于Monte Carlo模拟的IC模型，传播概率满足随机概率分布随机取值
def IC(G:nx.Graph, S:list, MC:int):
    """
    输入：G：社交网络图，S：种子集，p：传播概率，MC：Monte Carlo模拟的迭代次数
    输出：影响力扩展度(influence spread：种子集在每次迭代中所激活的节点数的期望值)
    """
    spread = []
    for _ in range(MC):
        # 遍历种子集中的所有结点
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for s in new_active:
                neighbors = list(G.neighbors(s))  # 得到种子s的所有邻居节点
                for neighbor in neighbors:
                    if(random.uniform(0, 1) < G[s][neighbor]['weight']):
                        new_ones.append(neighbor)

            new_active = set(new_ones) - set(A)
            A += new_active
        spread.append(len(A))
    return np.mean(spread)


if __name__ == "__main__":
    path = "./data/train/dataset-10/"
    graph = nx.read_weighted_edgelist(path + "edges.txt", create_using=nx.Graph(),nodetype=int)
    edges = list(graph.edges)
    for i in range(len(edges)):
        graph[edges[i][0]][edges[i][1]]['weight'] = 0.1
    
    print(IC(graph, [1, 3, 4, 7, 10], MC=10000))
  
    