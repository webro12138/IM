import os
import logging
from utils import get_logger
class Evalution:
    def __init__(self, metrics, networks, alg_list, x_range, x_range_type="number", logging_path=None):
        assert x_range_type in ["number", "percent"], "x_range_type值只能是'number'或'percent'"
        assert isinstance(x_range, list), "x_range必须是一个list"
        self._metrics = metrics
        self._networks = networks
        self._alg_list = alg_list
        self._x_range = x_range
        self._x_range_type = x_range_type
        self._logging_path = logging_path

    def __call__(self, retain_alg=None):
        if(self._logging_path != None):
            logger = get_logger("对比logging", self._logging_path)
        eva = {"x_label":"种子节点个数", "x_range":self._x_range, "x_range_type":self._x_range_type, "result":None}
        result = {}
        for metric in self._metrics:
            result[metric.get_name()] = {}
            for network in self._networks:
                result[metric.get_name()][network.get_name()] = {}
                for alg in self._alg_list:
                    result[metric.get_name()][network.get_name()][alg.get_name()] = []
                    if retain_alg and alg.get_name() in retain_alg:
                        for r in self._x_range:
                            if(self._x_range_type == "number"):
                                result[metric.get_name()][network.get_name()][alg.get_name()].append(metric(network, alg(network, r)))
                            else:
                                result[metric.get_name()][network.get_name()][alg.get_name()].append(metric(network, alg(network, int(network.number_of_nodes() * r))))
                            if(self._logging_path != None):
                                logger.info(f"在数据集上{network.get_name()},使用算法{alg.get_name()}"\
                                    f"选{r}个节点获得指标{metric.get_name()}的值为{ result[metric.get_name()][network.get_name()][alg.get_name()][-1]}")
                                
                            print(f"重计算方式>>在数据集上{network.get_name()},使用算法{alg.get_name()}"\
                                f"选{r}个节点获得指标{metric.get_name()}的值为{ result[metric.get_name()][network.get_name()][alg.get_name()][-1]}")
                    else:
                        if(self._x_range_type == "number"):
                            S = alg(network, self._x_range[-1])
                        else:
                            S = alg(network, int(network.number_of_nodes() * self._x_range[-1]))
                        
                        for r in self._x_range:
                            if(self._x_range_type == "number"):
                                result[metric.get_name()][network.get_name()][alg.get_name()].append(metric(network, S[:r]))
                            else:
                                result[metric.get_name()][network.get_name()][alg.get_name()].append(metric(network, S[:int(network.number_of_nodes() * r)]))
                                
                            if(self._logging_path != None):
                                logger.info(f"在数据集上{network.get_name()},使用算法{alg.get_name()}"\
                                    f"选{r}个节点获得指标{metric.get_name()}的值为{ result[metric.get_name()][network.get_name()][alg.get_name()][-1]}")
                                
                        print(f"重计算方式>>在数据集上{network.get_name()},使用算法{alg.get_name()}"\
                                f"选{r}个节点获得指标{metric.get_name()}的值为{ result[metric.get_name()][network.get_name()][alg.get_name()][-1]}")
        
        eva["result"] = result
        return eva


                            


            
