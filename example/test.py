import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from diffusionModel import ICWeighter, IC, SIRWeighter, SIR
from algorithm import DegreeCentrality, SAW_ASA, NCVoteRank, PageRank, DegreeDiscount, NDDAGCN, Greedy, CELF
from dataset import load_networks, load_features
from evalution import Evalution, InfluenceSpread
from classification_model.TFGCN import get_TFGCN
import networkx as nx
import numpy as np
from utils import save_json 
def load_TFGCN(filepath, input_dim):
    virtual_graph = nx.complete_graph(2)    
    model = get_TFGCN()
    model(np.zeros((2, input_dim)), virtual_graph)
    model.load_weights(filepath)
    return model

## 读取网络
# networks = load_networks(["filmtrust", "soc-wiki-Vote",  "aren-email", "stelzl","hamster-friend", "lastfm_asia", "CA-HepTh", "NetPHY"], "undirected")
networks = load_networks(["stelzl"], "undirected")

## 定义权重生成器
# weighter = SIRWeighter(APType="uniform", RPType="uniform", p=0.1, q=1)
# weighter.assign_activa_prob_batch(networks)
weighter = ICWeighter(APType="random", p=(0, 0.5))

## 定义IC模型
ic = IC(weighter=weighter, MC=1000, verbose=True)

## 定义degree中心性算法
dc = DegreeCentrality(verbose=True)

## 定义SAW_ASA算法
saw_asa = SAW_ASA(ic)

## 定义NCVoteRank算法
nc_vote_rank = NCVoteRank(0.25)

## 定义PageRank
pagerank = PageRank()

## 定义DegreeDiscount算法
degree_discount = DegreeDiscount()

## 定义NDD_AGCN算法
model = load_TFGCN(r"F:\wsw\python\JIAJIA\classification_model\model.h5", 9)

nddagcn = NDDAGCN(model = model, x=None)
changed_attr = {}
for network in networks:
    name = network.get_name()
    changed_attr[name] = {"x":load_features(name, network._graph)}
nddagcn.set_changed_attr_by_network(changed_attr)
## 定义Influence spread指标

IS = InfluenceSpread(ic, verbose=False)

## greedy
greedy = Greedy(diffusion_model=ic)

## CELF
celf = CELF(diffusion_model=ic)

## 使用IC模型衡量S的质量
ev = Evalution([IS], networks, [celf], list(range(5, 51, 5)), logging_path="IC_IS_.txt")
result = ev()
print(result)
save_json(result, "result_random_celf1.json")
