from network import Network
import networkx as nx
class SLN(Network):
    def __init__(self, graph):
        super().__init__()
        if(type(graph) == nx.Graph):
            self.gtype="indirected"
        if(type(graph) == nx.DiGraph):
            self.gtype="directed"
        self._graph = graph
        allnodes = self.nodes()
        
        self.index_of_nodes = {}
        for i in range(self.number_of_nodes()):
            self.index_of_nodes[allnodes[i]] = i 
        
    def number_of_edges(self):
        return self._graph.number_of_edges()
        
    def number_of_nodes(self):
        return self._graph.number_of_nodes()
    
    def edges(self):
        return list(self._graph.edges)
    
    def nodes(self):
        return list(self._graph.nodes)
    
    def get_node_index(self, v):
        """给定节点id获得节点位置索引
           输入:节点v
           输出:节点索引
        """
        return self.index_of_nodes[v]

    def get_index_from_node_banches(self, nodes):
    
        nodes_index = []
        for node in nodes:
            nodes_index.append(self.index_of_nodes[node])
        return nodes_index
    
    def __getitem__(self, key):
        return self._graph[key]

    def __setitem__(self, key, value):
        self._graph[key] = value
    
    def degrees(self, banch=None):
        degrees_dict = {}
        if(banch == None):
            banch = self.nodes()

        for node in banch:
            degrees_dict[node] = len(self._graph[node])
        
        return degrees_dict

    def neighbors(self, nodes):
        if(isinstance(nodes, list)):
            neis = []
            for node in nodes:
                neis += list(self._graph[node])
            return neis
        else:
            return self._graph[nodes]
    
    def set_node_attr(self, node, attr_name, value):
        self._graph.nodes[node][attr_name] = value
    
    def get_node_attr(self, node, attr_name):
        return self._graph.nodes[node][attr_name]
    
    def average_degree(self):
        return sum(dict(nx.degree(self._graph)).values()) / self.number_of_nodes()

    def pageRank(self, max_iters=600):
        return dict(nx.pagerank(self._graph, max_iter=max_iters))