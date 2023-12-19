from math import floor
from random import random
import networkx as nx
import random
import numpy as np
import os
from utils.load_data import write_to_file
from utils.processbar import gen_data_process_bar

def generate_dataset(graph_config, path):
    if(os.path.exists(path)):
        assert os.path.isdir(path), "path is not a directory" 
    else:
        os.mkdir(path)
    t = graph_config['type']
    assert t in ['er', 'ws', 'ba'], "graph type must be in ['er', 'ws', 'ba']"

    if(t == 'er'):
        n = graph_config['n']
        p = graph_config['p']
        seed = graph_config['seed']
        directed = graph_config['directed']
        G = nx.fast_gnp_random_graph(n, p, seed, directed)    
    elif(t == 'ws'):
        n = graph_config['n']
        p = graph_config['p']
        k = graph_config['k']
        seed = graph_config['seed']
        G = nx.watts_strogatz_graph(n, k, p, seed)
    else:    
        n = graph_config['n']
        m = graph_config['m']
        seed = graph_config['seed']
        init_graph = graph_config['init_graph']
        G = nx.barabasi_albert_graph(n,m,seed,init_graph)
    
    edge = list(G.edges)
    
    write_to_file(edge, path + "edges.txt")


if __name__ =="__main__":
    er_config = {
        'type':'er',
        'n':50,
        'p':0.05,
        'seed':None,
        'directed':None
    }
    
    ws_config = {
        'type':'ws',
        'n':50,
        'p':0.5,
        'k':30,
        'seed':None
    }
    ba_config = {
        'type':'ba',
        'n':50,
        'm':17,
        'seed':None,
        'init_graph':None
        
    }

   
   
    # for i in range(100, 200):
    #     ws_config['n'] = 50 * ((i - 100) + 1)
    #     ws_config['p'] = 6 * (i - 100 + 1)
    #     generate_dataset(ws_config, "./data/train/dataset-" + str(i) + "/" + "edges.txt")
    #     gen_data_process_bar(i - 100 + 1, 100, "dataset-" + str(i))
    
    
    # for i in range(600, 601):
        
    #     er_config['n'] = 50 * (floor(i / 5) + 1)
    #     er_config['p'] = 0.01   
    #     generate_dataset(er_config, "/home/webro/code/python/AGCNIM/data/synthesized_dataset/test_data/dataset-" + str(i) + "/")
        
    # for i in range(800, 801):   
    #     ws_config['n'] = 100 * (floor(i / 10)  + 1)
    #     ws_config['m'] = 100 * (floor(i / 10) + 1) * 0.34
    #     generate_dataset(ws_config, "/home/webro/code/python/AGCNIM/data/synthesized_dataset/test_data/dataset-" + str(i) + "/")
        
    for i in range(0, 300):
        ba_config['n'] = 50 * (floor(i / 5) + 1)
        ba_config['m'] =  int((4 / (50 + i * 5)) * ba_config['n'])
        generate_dataset(ba_config, "/home/webro/code/python/agcnim/data/synthesized_dataset/train_data3/dataset-" + str(i) + "/")
     
        
    #     gen_data_process_bar(i + 1, 100, "dataset-" + str(i))
       
        
    
