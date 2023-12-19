from typing import Any
import networkx as nx
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
class SIR:
    """This is a class of classical infectious disease models.
    
    Attributes: 
        p: A float Indicates the probability that nodes in I state will be activated by nodes in S state
        q: A float Indicates the possibility of a node in state I reverting to a node in state R
    """
    def __init__(self, p:float, q:float, MC:int) -> None:
        self.p = p
        self.q = q
        self.MC = MC
    def step(self, network:nx.Graph, S:list[int])->float:
        S = S[:]
        R = []
        
        t = 0
        new_S = [0]
        while len(new_S) > 0:
            new_S = []
            new_R = []
            for s in S:
                
                neis = list(set(nx.neighbors(network, s)) - set(S) - set(R))   
         
                for nei in neis:
                    if(self.p > random.uniform(0,1)):
                        new_S.append(nei)
                    
                if(self.q > random.uniform(0, 1)):
                    R.append(s)
                    new_R.append(s)
                     
            
            S += new_S
            
            S = list(set(S) - set(new_R))
            t += 1
           
        return len(S) + len(R)
                    
    def simulate(self, network:nx.Graph, S:list[any]) -> float:
        
        loops = tqdm(range(self.MC))
        IS = 0
        for _ in loops:
            IS += self.step(network, S)
        IS /= self.MC
        
        return IS
    
    def __call__(self, network: nx.Graph, S: list) -> Any:
        return self.simulate(network, S)