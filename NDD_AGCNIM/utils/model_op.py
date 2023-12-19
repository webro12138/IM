from model.classification_model.TFGCN import get_TFGCN
import networkx as nx
import numpy as np
def load_TFGCN(filepath, input_dim):
    virtual_graph = nx.complete_graph(2)    
    model = get_TFGCN()
    model(np.zeros((2, input_dim)), virtual_graph)
    model.load_weights(filepath)
    return model