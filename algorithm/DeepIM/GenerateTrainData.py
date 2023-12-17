import networkx as nx
import torch
import random
import pickle
from tqdm import tqdm
from scipy.sparse import csr_matrix
import os
class GenerateTrainData:
    def __init__(self,diffusion_model, number_of_samples, data_file=None) -> None:
        self.diffusion_model = diffusion_model
        self.diffusion_model.set_is_return_node_ap(True)
        self.diffusion_model.set_verbose(False)
        self.number_of_samples = number_of_samples
        self.data_file = data_file

        
    def __call__(self, network, k):
        
        graph = self.run(network, k)
        self.diffusion_model.set_verbose(True)
        self.diffusion_model.set_is_return_node_ap(False)
        return graph
    
    def set_setting(self, network, k):
        self._network = network
        self._k = k
        
    def run(self, network, k):
 
        self.set_setting(network, k)
        
        if(self.data_file != None):
            if(os.path.exists(self.data_file)):
                with open(self.data_file, "rb") as f:
                    graph = pickle.load(f)
                return graph
        
        
        graph = {}
        adj_matrix = nx.adjacency_matrix(self._network._graph)
        graph["adj"] = csr_matrix(adj_matrix)
        self._seed_sets = self.sample()
        graph["inverse_pairs"] = self.construct_inverse_pairs()  
        
        if(self.data_file !=None):
            with open(self.data_file,"wb") as f:
                pickle.dump(graph,f)
            
        return graph
    
    def construct_inverse_pairs(self):
        inverse_pair = torch.zeros(size=(self.number_of_samples, self._network.number_of_nodes(), 2))
        
        loops = tqdm(range(self.number_of_samples))
        for i in loops:
            _, ap = self.diffusion_model(self._network, self._seed_sets[i])
            inverse_pair[i,self._network.get_index_from_node_banches(self._seed_sets[i]),0] = 1
            
            for node in ap:
                inverse_pair[i, self._network.get_node_index(node) ,1] = ap[node]
        
        return inverse_pair
                

    def sample(self):
        seed_sets = [random.sample(self._network.nodes(), self._k) for _ in range(self.number_of_samples)]
        return seed_sets
        