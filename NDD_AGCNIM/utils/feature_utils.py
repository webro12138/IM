import networkx as nx


def extract_network_attribute(G:nx.Graph):
    attribution = [] 
    attribution.append(len(G.nodes))
    attribution.append(len(G.edges))
    attribution.append(average_degree(G))
    attribution.append(nx.cluster.average_clustering(G))
    return attribution

def average_degree(G:nx.Graph):
    allnode = list(G.nodes)
    average = 0
    
    for node in allnode:
        average = average + nx.degree(G, node)
    
    return average / len(allnode)
