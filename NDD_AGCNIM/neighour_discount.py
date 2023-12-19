from unittest import result
import networkx as nx
import math

def neighour_discount(G, k, theta=3):
    degrees = dict(nx.degree(G))
    S = []
    history = {}
    for _ in range(k):

        s = max(degrees.items(), key =lambda x:x[1])[0]
        
        nei = list(set(G[s]) - set(S))
        for node in set(nei):
            if(node in history):
                ##if(len(set(nei) & set(G[node])) - history[node] > 0):
                degrees[node] -= len(set(nei) & set(G[node]) - history[node])
                history[node] = history[node] | (set(nei) & set(G[node]))
            else:
                degrees[node] -= len(set(nei) & set(G[node]))
                history[node] = set(nei) & set(G[node])
            
        degrees.pop(s)
        S.append(s)

    return S