from algorithm import AlgorithmBase
import xgboost as xgb
import numpy as np
class XgboostIM(AlgorithmBase):
    def __init__(self, features, model_file, verbose=True):
        super().__init__()
        self._verbose = verbose
        self.features = features
        self.model_file = model_file
    
    def run(self, network, k):
        xgb_model = xgb.Booster(model_file = self.model_file)
        data = xgb.DMatrix(list(self.features.values()))    
        pred = xgb_model.predict(data)
        max_pred = max(pred)
        S = []
        nodes = list(self.features.keys())
        for _ in range(k):
            s = np.argmax(pred)
            pred[s] = -1
            S.append(nodes[s])
        return S

