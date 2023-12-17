from diffusionModel import DiffusionBase
from diffusionModel import Weighter
import random
import network as nx
from tqdm import tqdm
class SIRWeighter(Weighter):

    def __init__(self, APType="random", RPType="random", p=None, q=None):
        super(Weighter, self)
        assert APType in ["uniform", "random"], "APType的值只能时'random'或'uniform'"
        assert RPType in ["uniform", "random"], "RPType的值只能时'random'或'uniform'"

        self._APType = APType
        self._RPType = RPType
        self._p = p
        self._q = q

    def assign_active_prob(self, network):
            if(self._APType == "random"):
                for node1, node2 in network.edges():
                    network[node1][node2]["weight"] = random.uniform(self._p[0], self._p[1])
            if(self._APType == "uniform"):
                for node1, node2 in network.edges():
                    network[node1][node2]["weight"] = self._p
            
            if(self._RPType == "random"):
                for node in network:
                    network._graph.nodes[node]["recovery"] = random.uniform(self._q[0], self._q[1])
            if(self._RPType == "uniform"):
                for node in network.nodes():
                    network._graph.nodes[node]["recovery"] = self._q
            network.set_type_of_weight(self.__class__.__name__)
    def get_active_prob(self, network, edge):
        return network[edge[0]][edge[1]]["weight"] 

    def get_recovery_prob(self, network, node):
        return network._graph.nodes[node]["recovery"] 

class SIR(DiffusionBase):
    """This is a class of classical infectious disease models.

    """
    def __init__(self, weighter, MC:int, is_return_node_ap=False, verbose=True) -> None:
        super().__init__()
        self._weighter = weighter
        self.MC = MC
        self._verbose = verbose
        self.is_return_node_ap = is_return_node_ap
    
    def step(self, network, S:list):
        S = S[:]
        R = []
        t = 0
        new_S = [0]
        while len(new_S) > 0:
            new_S = []
            new_R = []
            for s in S:
                
                neis = list(set(network[s]) - set(S) - set(R))   
         
                for nei in neis:
                    if(self._weighter.get_active_prob(self._network, (s, nei)) > random.uniform(0,1)):
                        new_S.append(nei)
                    
                if(self._weighter.get_recovery_prob(self._network, s) > random.uniform(0, 1)):
                    R.append(s)
                    new_R.append(s)
                     
            
            S += new_S
            
            S = list(set(S) - set(new_R))
            t += 1
        if self.is_return_node_ap:
            return len(S) + len(R), list(set(S) & set(R))
        else:
            return len(S) + len(R)
                    
    def simulate(self, network, S:list[any]) -> float:
        self.set_network(network)
        self.set_S(S)
        self._weighter.assign_weights(self._network)

        if(self._verbose):
            loops = tqdm(range(self.MC))
        else:
            loops = range(self.MC)
        IS = 0
        node_ap = {}
        for _ in loops:
            if self.is_return_node_ap:
                temp1, temp2 = self.step(self._network, self._S)
                for node in temp2:
                    node_ap[node] = node_ap.setdefault(node, 0) + 1
                IS += temp1
            else:
                IS += self.step(self._network, self._S)
        
        if self.is_return_node_ap:
            for node in node_ap:
                node_ap[node] /= self.MC
            IS = IS / self.MC
            return IS, node_ap
        else:
            IS = IS / self.MC
            return IS
    
    def __call__(self, network, S: list):
        return self.simulate(network, S)

    def inactive_probability(self, node):
        if(self._network.gtype=="directed"):
            neighbors = list(self._network.predecessors(node))
        else:
            neighbors = list(self._network[node])
        prob = 1.0
        for neighbor in neighbors:
            if(neighbor in self._S):
                weight = self._network[neighbor][node]["weight"]
                prob *= (1 - weight)
        return prob

    def approx_func(self, network, S):
        self.set_network(network)
        self.set_S(S)
        self._weighter.assign_weights(self._network)
        neighbors = self._network.neighbors(S)
        
        edv = len(self._S)
        for node in self._network.nodes():
            if node not in self._S and node in neighbors:
                edv += 1 - self.inactive_probability(node)
        return edv
    
    def set_is_return_node_ap(self, value):
        self.is_return_node_ap = value