import networkx as nx
import math

def classifiy_strong_node(degrees:dict):
    
    degrees_sort = sorted(degrees.items(), key=lambda x:x[1])  
    frequent = {}
    for key in degrees_sort:
        frequent[key[1]] = frequent.setdefault(key[1], 0) + 1  
    
    cur_point = 0
    values = list(frequent.values())
    degrees_list = list(frequent.keys())
    diff = []
    for i in range(len(values) - 1):
        diff.append(math.fabs(values[i + 1] - values[i]))
        if(math.fabs(values[i + 1] - values[i]) <= 1):
            flag = True
            for j in range(i, i + 3):
                if(j + 1 < len(values)):
                    if(math.fabs(values[j + 1] - values[j]) > 1):
                        flag = False
        
            if(flag):
                cur_point = degrees_list[i]
                break       
    influntial_nodes = []
    degrees = dict(degrees)
    for i in degrees:
        if(cur_point < degrees[i]):
            influntial_nodes.append(i)  

    return influntial_nodes, cur_point

def get_labels(G, k):
    result = {}
    degrees = dict(nx.degree(G))
    
    
    history = {}
    S = []
    for _ in range(k):
        #if(influntial_nodes == []):
        s = max(degrees.items(), key =lambda x:x[1])[0] 
        nei = list(set(G[s]) - set(S))
        for node in nei:
            if(node in history):
                ##if(len(set(nei) & set(G[node])) - history[node] > 0):
                degrees[node] -= len(set(nei) & set(G[node]) - history[node])
                history[node] = history[node] | (set(nei) & set(G[node]))
            else:
                degrees[node] -= len(set(nei) & set(G[node]))
                history[node] = set(nei) & set(G[node])
        result[s] = degrees[s]
        degrees.pop(s)
        S.append(s)
    max_value = max(result.items(), key =lambda x:x[1])[1]
    min_value = min(result.items(), key =lambda x:x[1])[1]
        
    for key in result:
        result[key] = (result[key] - min_value) / (max_value - min_value)
  
    return result


def get_labels_by_pagerank(G, k):
    pr = dict(nx.pagerank(G))
    return pr

def get_labels_by_degree(G, k):
    de = dict(nx.degree(G))
    return de