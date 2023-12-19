from math import floor
from utils.load_data import read_dict, write_to_file
import networkx as nx
import numpy as np
# from ge import DeepWalk
from utils.feature_utils import extract_network_attribute

class Dataset:
    def __init__(self):
       self.graph = None
       self.features = None
       self.labels = None  
    def load_features(self, type="cal", path = None):
        """
            type决定features是从目录读取，还是重新计算。
        """    
        
        
        if(type == 'cal'):
            assert self.graph != None, "请先load graph"
            self.features = self.extract_features()  
        
        if(type == 'load'):
            self.features = read_dict(path)
        
    
    def load_graph(self, path, weighted = False, weights = None):
        self.graph = nx.read_weighted_edgelist(path, create_using=nx.Graph())
        
        if(weighted):
                allEdges = list(self.graph.edges)
                if(isinstance(weights, (int, np.long, float))):
                    for i in range(len(allEdges)):
                        self.graph[allEdges[i][0]][allEdges[i][1]]['weight']= weights
                if(type(weights) == list):
                    for i in range(len(allEdges)):
                        self.graph[allEdges[i][0]][allEdges[i][1]]['weight']= weights[i]
    
    def set_weight(self, weights):
        allEdges = list(self.graph.edges)
        if(isinstance(weights, (int, float))):
            for i in range(len(allEdges)):
                self.graph[allEdges[i][0]][allEdges[i][1]]['weight']= weights
        if(type(weights) == list):
            for i in range(len(allEdges)):
                self.graph[allEdges[i][0]][allEdges[i][1]]['weight']= weights[i]
    
    
    def save_features(self, path):
        assert self.features != None, "请先load features"
        write_to_file(self.features, path)
        
    def get_graph(self):
        if(self.graph == None):
            self.load_graph()
        return self.graph
    
    
    def get_features_list(self):
        if(self.graph == None):
            self.load_graph()
        
        if(self.features == None):
            self.load_features()
        
        features = []
        
        for i in self.graph.nodes:
        
            features.append(self.features[i].tolist())
        return features
    
    
    def load_labels(self, method, type="cal", path=None):
        if(self.graph == None):
            self.load_graph()
        if(type == "cal"):
            labels = method(self.graph, floor(len(list(self.graph.nodes))))
            
            # allNodes = list(self.graph.nodes)
            # labels = {}
            # for i in allNodes:
            #     if(i in S):
            #         labels[i] = 1
            #     else:
            #         labels[i] = 0
            labels = sorted(labels.items(), key=lambda x:x[0], reverse=False)
            self.labels = dict(labels)
        if(type=="load"):
            self.labels = read_dict(path)
            
    
    
    def get_labels_list(self):
        labels = []
        allNodes = list(self.graph.nodes)
        
        for i in allNodes:
            labels.append(self.labels[i])
        return labels

    def save_lables(self, path):
        assert self.labels != None, "请先load features"
        write_to_file(self.labels, path)
        
        
    def extract_features(self, demension = 9):
        # model = DeepWalk(self.graph, walk_length=10,num_walks=80,workers=1)
        # model.train()
        # features = model.get_embeddings()
        # features = []
        features = {}
        attribute = extract_network_attribute(self.graph)
        allNode = list(self.graph.nodes)
        allNode.sort()
        
        bc = nx.centrality.betweenness_centrality(self.graph)
        dc = nx.centrality.degree_centrality(self.graph)
        cc = nx.centrality.closeness_centrality(self.graph)
        ec = nx.centrality.eigenvector_centrality(self.graph,  max_iter=600)
        pr = nx.algorithms.pagerank(self.graph, alpha=0.9)
        
        for node in allNode:
            feature = np.zeros(9)
            for j in range(len(attribute)):
                feature[j] = attribute[j]

            feature[4] = dc[node]
            feature[5] = bc[node]
            feature[6] = cc[node]
            feature[7] = ec[node]
            feature[8] = pr[node]
            features[node] = feature   
                
        
        return features
            
