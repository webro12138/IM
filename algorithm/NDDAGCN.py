from algorithm import AlgorithmBase
class NDDAGCN(AlgorithmBase):
    def __init__(self, model, x, verbose=True):
        super().__init__()
        self._verbose = verbose
        self.model = model
        self.x = x
    
    def run(self, network, k):
        S = self.model.find_seeds(self.x, network._graph, k)
        return S
